import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F

from hidden_state_cache import HiddenStateCache

from transformers.models.mpt.modeling_mpt import (
    MptAttention,
    MptBlock,
    MptForCausalLM,
    MptModel,
)
import types

__all__ = ["enable_mpt_masked_cache"]

# pass in hidden_state_cache, up to user to update the hidden_state_cache


def mpt_attention_forward(  # MptAttention
    self,
    hidden_states: torch.Tensor,
    position_bias: torch.Tensor,
    # past_key_value: Optional[Tuple[torch.Tensor]] = None,
    # past_key_value_mask: Optional[torch.Tensor] = None,
    past_hidden_states: Optional[torch.Tensor] = None,  # batchxseq_lengthxhidden_size
    attention_mask: Optional[torch.Tensor] = None,
):
    batch_size, seq_length = hidden_states.shape[:2]

    mixed_qkv = self.Wqkv(hidden_states)
    query_states, key_states, value_states = mixed_qkv.chunk(3, dim=2)
    query_states = query_states.reshape(
        batch_size, seq_length, self.n_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.reshape(
        batch_size, seq_length, self.n_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.reshape(
        batch_size, seq_length, self.n_heads, self.head_dim
    ).transpose(1, 2)

    # if past_key_value is not None:
    #     if len(past_key_value) != 0:
    #         key_states = torch.cat([past_key_value[0], key_states], dim=2)
    #         value_states = torch.cat([past_key_value[1], value_states], dim=2)
    #     past_key_value = (key_states, value_states)
    # else:
    #     past_key_value = (key_states, value_states)

    if past_hidden_states is not None:  ####
        if past_hidden_states.shape[2] != 0:
            past_batch_size, past_seq_length = past_hidden_states.shape[:2]
            past_qkv = self.Wqkv(past_hidden_states)
            past_query_states, past_key_states, past_value_states = past_qkv.chunk(
                3, dim=2
            )
            past_key_states = past_key_states.reshape(
                past_batch_size, past_seq_length, self.n_heads, self.head_dim
            ).transpose(1, 2)
            past_value_states = past_value_states.reshape(
                past_batch_size, past_seq_length, self.n_heads, self.head_dim
            ).transpose(1, 2)
            key_states = torch.cat([past_key_states, key_states], dim=2)
            value_states = torch.cat([past_value_states, value_states], dim=2)
        past_key_value = (key_states, value_states)
    else:
        past_key_value = (key_states, value_states)

    attention_scores = (
        torch.matmul(query_states, key_states.transpose(-1, -2)) * self.softmax_scale
    )
    ### query_states shape: [batch_size, n_heads, seq_length, head_dim]
    ### key_states shape: [batch_size, n_heads, head_dim, past_key_value_length + seq_length]
    ### Anyways, this automatically recognizes the first dimension are the same, so it will broadcast to last 2!!
    ### We are doing batch_sizexn_heads matrix multiplication of seq_lengthxhead_dim * (past_key_value_length + seq_length)
    ### attention_scores shape: [batch_size, n_heads, seq_length, past_key_value_length + seq_length]

    query_length = (
        seq_length
        if past_hidden_states is None  ####
        else seq_length + past_hidden_states.shape[2]
    )

    if position_bias is not None:
        if len(position_bias.shape) != 3:
            raise ValueError(
                f"Expecting position_bias shape to be 3 dimensions, got {len(position_bias.shape)}"
            )
        key_length = key_states.shape[-2]

        position_bias_query_index = max(0, position_bias.size(1) - query_length)
        position_bias_key_index = max(0, position_bias.size(2) - key_length)

        position_bias = position_bias[
            :, position_bias_query_index:, position_bias_key_index:
        ]

        attention_scores = attention_scores + position_bias

    if attention_mask is not None:
        attention_scores = attention_scores.masked_fill(
            attention_mask, torch.finfo(query_states.dtype).min
        )
    ### attention_mask is shape [batch_size, 1, seq_length, past_key_value_length + seq_length] according to the code
    ### in the function definition they say shape [batch_size, 1, query_length, key_value_length]
    ### it can broadcast across the 1!!

    # (batch_size, n_heads, seq_length, key_length)
    attn_weights = nn.functional.softmax(attention_scores.float(), dim=-1).to(
        value_states.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attn_dropout_p, training=self.training
    )

    context_states = torch.matmul(attn_weights, value_states)
    context_states = (
        context_states.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
    )
    attn_output = self.out_proj(context_states)

    return attn_output, attn_weights, past_key_value


def mpt_block_forward(  # MptBlock
    self,
    hidden_states: torch.Tensor,
    position_bias: torch.Tensor,
    attention_mask: torch.Tensor,
    # layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    # layer_past_mask: Optional[torch.Tensor] = None,
    layer_past_hidden_states: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
):
    # hidden_states: [batch_size, seq_length, hidden_size]
    # Layer norm at the beginning of the transformer layer.
    layernorm_output = self.norm_1(hidden_states)

    residual = hidden_states

    # Self attention.
    attn_outputs, attn_weights, past_key_value = self.attn(
        layernorm_output,
        position_bias=position_bias,
        attention_mask=attention_mask,
        # past_key_value=layer_past,
        # past_key_value_mask=layer_past_mask,
        past_hidden_states=layer_past_hidden_states,
    )

    hidden_states = self.resid_attn_dropout(attn_outputs) + residual

    layernorm_output = self.norm_2(hidden_states)

    # Get residual
    residual = hidden_states

    # MLP.
    output = self.ffn(layernorm_output, residual)
    outputs = (output,)

    if use_cache:
        outputs += (past_key_value,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs  # hidden_states, present, attentions


def mpt_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    # past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    # past_key_value_mask: Optional[Tuple[torch.Tensor, ...]] = None,  ###
    hidden_state_cache: Optional[HiddenStateCache] = None,
    attention_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    # if past_key_values is None:
    #     past_key_values = tuple([None] * len(self.blocks))

    # if past_key_value_mask is None:  ### Added this
    #     past_key_value_mask = tuple([None] * len(self.blocks))

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

    hidden_states = inputs_embeds

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # Compute alibi tensor: check build_alibi_tensor documentation
    seq_length_with_past = seq_length
    # past_key_values_length = 0
    # if past_key_values[0] is not None:
    #     past_key_values_length = past_key_values[0][0].shape[2]
    #     seq_length_with_past = seq_length_with_past + past_key_values_length
    past_hidden_states_length = hidden_state_cache.get_past_length()
    seq_length_with_past = seq_length_with_past + past_hidden_states_length
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), device=hidden_states.device
        )
    else:
        attention_mask = attention_mask.to(hidden_states.device)

    alibi = self.build_mpt_alibi_tensor(
        self.num_heads, self.config.max_seq_len, device=hidden_states.device
    )

    causal_mask = _prepare_4d_causal_attention_mask(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_hidden_states_length,
        # past_key_values_length,
    )
    causal_mask = causal_mask.bool()

    for i, block in enumerate(self.blocks):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(
                block.__call__,
                hidden_states,
                alibi,
                causal_mask,
                None,  # layer_past,
                use_cache,
                output_attentions,
            )
        else:
            outputs = block(
                hidden_states,
                # layer_past=layer_past,  ### past_key_values passed in here
                # layer_past_mask=layer_past_mask,  ### past_key_value_mask passed in here
                layer_past_hidden_states=self.hidden_state_cache.get_layer(i),
                attention_mask=causal_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                position_bias=alibi,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (
                outputs[2 if use_cache else 1],
            )

    # Add last hidden state
    hidden_states = self.norm_f(hidden_states)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                presents,
                all_hidden_states,
                all_self_attentions,
            ]
            if v is not None
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def causal_lm_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    # past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    # past_key_value_mask: Optional[Tuple[torch.Tensor, ...]] = None,
    hidden_state_cache: Optional[HiddenStateCache] = None,
    attention_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
        are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
    """
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    transformer_outputs = self.transformer(
        input_ids,
        # past_key_values=past_key_values,
        # past_key_value_mask=past_key_value_mask,
        hidden_state_cache=hidden_state_cache,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = transformer_outputs[0]

    lm_logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        batch_size, seq_length, vocab_size = shift_logits.shape
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(batch_size * seq_length, vocab_size),
            shift_labels.view(batch_size * seq_length),
        )

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithCrossAttentions(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )


def enable_mpt_hidden_state_cache(model):
    model.forward = types.MethodType(causal_lm_forward, model)
    model.transformer.forward = types.MethodType(mpt_model_forward, model.transformer)

    for name, module in reversed(model._modules.items()):  # why do we reverse it?
        if len(list(module.children())) > 0:
            enable_mpt_hidden_state_cache(
                module,
            )

        if isinstance(module, MptAttention):
            model._modules[name].forward = types.MethodType(
                mpt_attention_forward, model._modules[name]
            )

        if isinstance(module, MptBlock):
            model._modules[name].forward = types.MethodType(
                mpt_block_forward, model._modules[name]
            )
