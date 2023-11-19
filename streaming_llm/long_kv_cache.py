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

### If we do sparse token retrieval, do we need to mess with position_ids?
### For MPT, we pass in num_layer * [batch_size, n_heads, past_key_value_length]
### past_key_values is num_layer * ([batch_size * num_heads, head_dim, kv_length], [batch_size * num_heads, kv_length, head_dim]) according to docs
### Wait this is kinda sus because according to the code, K,V are both shape [batch_size, seq_length, self.head_dim, self.n_heads]
### Is the documentation wrong? Lets just assume the doc is wrong, and the code is right...

# D: Drawback is we now need to make sure the num_batch and stuff all line up!!
# D: Also bad thing is all this slicing will make it fragmented in memory!!


class LongKVCache:
    def __init__(
        self,
        batch_size,
        n_layers,
        n_heads,
        start_size=4,  # D: number of sink tokens
        cache_size=512,
        k_seq_dim=1,
        v_seq_dim=1,
        layer_head_mask=None,  # shape [n_layers, n_heads]
    ):
        print(f"StartLongKVCache: {cache_size}")
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.start_size = start_size
        self.cache_size = cache_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]  # D: function like slice2d
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

        self.key_values = None
        self.sink_key_values = None

        if layer_head_mask is None:
            self.layer_head_mask = torch.ones((self.n_layers, self.n_heads))
            # this mask lets us mask at the layer/head level
        else:
            self.layer_head_mask = layer_head_mask

    def add_recent_to_cache(self, incoming_key_values, num_incoming):
        """
        Only add the most recent num_incoming of incoming_key_values
        """

        incoming_key_values = [
            [
                self.k_slice(k, -num_incoming, None),
                self.v_slice(v, -num_incoming, None),
            ]
            for k, v in incoming_key_values
        ]

        self.__call__(incoming_key_values)

    def __call__(self, incoming_key_values=None):
        """
        We add past_key_values to the cache, and evict if we go over self.cache_size
        """
        if incoming_key_values is None:
            return self.key_values
        seq_len = incoming_key_values[0][0].size(self.k_seq_dim)
        past_kv_len = self.key_values[0][0].size(self.k_seq_dim)

        if not self.key_values:
            self.key_values = incoming_key_values  # lazy initialization lmfao
        else:
            self.key_values = [
                [
                    torch.cat(
                        [
                            self.key_values[index][0],
                            k,
                        ],
                        dim=self.k_seq_dim,
                    ),
                    torch.cat(
                        [
                            self.key_values[index][1],
                            v,
                        ],
                        dim=self.v_seq_dim,
                    ),
                ]
                for index, (k, v) in enumerate(zip(incoming_key_values))
            ]

        if seq_len + past_kv_len > self.cache_size:
            self.key_values = [
                [
                    torch.cat(
                        [
                            self.k_slice(k, 0, self.start_size),
                            self.k_slice(
                                k,
                                seq_len - self.cache_size + self.start_size,
                                seq_len,
                            ),
                        ],
                        dim=self.k_seq_dim,
                    ),
                    torch.cat(
                        [
                            self.k_slice(v, 0, self.start_size),
                            self.k_slice(
                                v,
                                seq_len - self.cache_size + self.start_size,
                                seq_len,
                            ),
                        ],
                        dim=self.v_seq_dim,
                    ),
                ]
                for k, v in self.key_values
            ]

        return self.key_values

    def get_kv_mask(self, recent_size=50):
        seq_len = self.key_values[0][0].size(self.k_seq_dim)
        mask = []

        for i in range(self.n_layers):
            if seq_len > recent_size + self.start_size:
                mask_i = torch.cat(
                    torch.ones(self.batch_size, self.n_heads, self.start_size),
                    self.layer_head_mask[i]
                    .reshape(1, -1, 1)
                    .repeat(
                        self.batch_size, 1, seq_len - recent_size - self.start_size
                    ),
                    # For now we just have same mask for all old entries
                    torch.ones(self.batch_size, self.n_heads, recent_size),
                    dim=2,
                )
            else:
                mask_i = torch.ones(self.batch_size, self.n_heads, seq_len)

            ### TODO: change the shape of mask_i to be layer_head_mask shape + an extra dimension seq_len

            mask.append(mask_i)

        mask = tuple(mask)

        mask = torch.ones(self.batch_size, self.n_heads, recent_size)

        return self.key_values, mask
