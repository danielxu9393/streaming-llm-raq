import torch


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class HiddenStateCache:
    def __init__(
        self,
        batch_size,
        n_layers,
        keep_hidden_layer_idx=None,
        seq_dim=1,
        start_size=4,  # D: number of sink tokens
        recent_size=50,
        cache_size=512,
    ):
        print(f"StartLongKVCache: {cache_size}")
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.start_size = start_size
        self.cache_size = cache_size
        self.recent_size = recent_size
        self.seq_slice = DIM_TO_SLICE[seq_dim]

        self.hidden_states = None

        if keep_hidden_layer_idx is None:
            self.keep_hidden_layer_idx = [i for i in range(n_layers)]
        else:
            self.keep_hidden_layer_idx = sorted(keep_hidden_layer_idx)

        self.layer_replace_idx = []
        j = 0
        for i in self.keep_hidden_layer_idx:
            if i >= j:
                self.layer_replace_idx.append(i)
                j += 1

    def slice_hidden_states(self, layer_hidden_state, layer_idx):
        if layer_idx in self.keep_hidden_layer_idx:
            max_len = self.cache_size
        else:
            max_len = self.recent_size

        if layer_hidden_state.size(self.seq_dim) <= max_len:
            return layer_hidden_state
        else:
            return torch.cat(
                [
                    self.seq_slice(layer_hidden_state, 0, self.start_size),
                    self.seq_slice(
                        layer_hidden_state,
                        layer_hidden_state.size(self.seq_dim) - max_len,
                        layer_hidden_state.size(self.seq_dim),
                    ),
                ],
                dim=self.seq_dim,
            )

    def add_to_cache(self, incoming_hidden_states):
        """
        We add hidden_states to the cache, and evict if we go over self.cache_size

        incoming_hidden_states: tuple[torch.Tensor], n_layers+1 * [batch_size, seq_length, hidden_size]
        """
        seq_len = incoming_hidden_states[0].size(self.seq_dim)
        past_kv_len = self.key_values[0][0].size(self.k_seq_dim)

        if not self.hidden_states:
            self.hidden_states = [
                self.slice_hidden_states(incoming_hidden_layer, layer_idx)
                for layer_idx, incoming_hidden_layer in enumerate(
                    incoming_hidden_states[: self.n_layers]
                )
            ]

        else:
            self.hidden_states = [
                self.slice_hidden_states(
                    torch.cat(
                        [
                            incoming_hidden_layer,
                            past_hidden_layer,
                        ],
                        dim=self.seq_dim,
                    ),
                    layer_idx,
                )
                for layer_idx, (incoming_hidden_layer, past_hidden_layer) in enumerate(
                    zip(incoming_hidden_states[:self], self.hidden_states)
                )
            ]

    def get_layer(self, layer_idx):
        cache_seq_len = self.hidden_states[self.layer_replace_idx[0]].size(self.seq_dim)
        # max number in cache already
        if (
            layer_idx in self.keep_hidden_layer_idx
            or cache_seq_len <= self.recent_size + self.start_size
        ):
            return self.hidden_states[layer_idx]
        else:
            assert (
                self.hidden_states[layer_idx].size(self.seq_dim)
                == self.recent_size + self.start_size
            )
            return torch.cat(
                [
                    self.seq_slice(
                        self.hidden_states[layer_idx],
                        0,
                        self.start_size,
                    ),
                    self.seq_slice(
                        self.hidden_states[self.layer_replace_idx[layer_idx]],
                        self.start_size,
                        cache_seq_len - self.recent_size,
                    ),
                    self.seq_slice(
                        self.hidden_states[layer_idx], self.start_size, self.recent_size
                    ),
                ],
                dim=self.seq_dim,
            )

    def get_past_length(self):
        return self.hidden_states[self.layer_replace_idx[0]].size(self.seq_dim)
