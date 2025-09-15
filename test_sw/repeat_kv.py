import torch


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        print(x)
        x = x[:, :, :, None, :]
        print(x)
        x = x.expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        print(x)
        x = x.reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
        print(x)
        return x


x = torch.tensor(
    [
        [
            [[1, 2, 3], [4, 5, 6]],  # 第一个 token 的两个 KV 头
            [[7, 8, 9], [10, 11, 12]],  # 第二个 token 的两个 KV 头
        ]
    ],
    dtype=torch.float32,
)

x_repeated = repeat_kv(x, n_rep=2)
print("x_repeated shape:", x_repeated.shape)
print(x_repeated)


a = torch.tensor(
    [
        [
            [[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]]],
            [
                [[7.0, 8.0, 9.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [10.0, 11.0, 12.0]],
            ],
        ]
    ]
)


tensor(
    [
        [
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [10.0, 11.0, 12.0]],
        ]
    ]
)
