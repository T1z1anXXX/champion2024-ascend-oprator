# champion2024-ascend-oprator
| Shape (参数的形状) | TypeRangeAll (设计规格：算子支持的数据类型) | Attr_Default_value | Format类型 | 参考算子&算子输入说明 | 参考资料 |
|-------------------|---------------------------------------------|--------------------|------------|-----------------------|----------|
| (N)               | float16, bfloat16, float32, int8, int32, uint8 |                    | ND         | 参考算子：<br>torch.trunc | [torch.trunc](https://docs.pytorch.org/docs/2.1/generated/torch.trunc.html) |
| (N, N2, N3)       | float16, bfloat16, float32, int8, int32, uint8 |                    | ND         | 算子输入shape范围 [...,N4,N3,N2,N]：<br>N∈[128, 2048] |          |
| (N, N2, N3, N4)   | float16, bfloat16, float32, int8, int32, uint8 |                    | ND         | N2∈[1,1024]<br>N3∈[1, 64]<br>N4∈[1,8] |          |
| (N, N2)           | float64, float32, float16, int8, uint8, int16, uint16, int32, uint32, int64, uint64, bool, bfloat16 |                    | ND         | 参考算子：<br>y = torch.nonzero(x)<br>if transpose:<br>    y = y.transpose() | [torch.nonzero](https://docs.pytorch.org/docs/2.1/generated/torch.nonzero.html#torch-nonzero) |
| (N, N2, N3)       | float64, float32, float16, int8, uint8, int16, uint16, int32, uint32, uint64, int64, bool, bfloat16 | FALSE              |            | if dtype == 3:<br>    y = y.astype(np.int32)<br>else:<br>    y = y.astype(np.int64) |          |
| /                 | 3, 9 (3为int32, 9为int64)                     | 9                  |            |                       |          |
| (N)               | int64, int32                                 |                    | ND         |                       |          |
| (N, N2)           | int64, int32                                 |                    | ND         |                       |          |
| (N, N2, N3)       | int64, int32                                 |                    | ND         |                       |          |
| [1]               | float32, float64, int8, uint8, int32, int16, float16, bfloat16 |                    | ND         | 参考算子：<br>torch.linspace | [torch.linspace](https://docs.pytorch.org/docs/2.1/generated/torch.linspace.html#torch-linspace) |
| [1]               | float32, float64, int8, uint8, int32, int16, float16, bfloat16 |                    | ND         | 算子输入shape范围：<br>start∈[-4096, 4096]<br>stop∈[-4096, 4096]<br>num_axes∈[32,2097152] |          |
| [num_axes[0]]     | float32, float64, uint8, int8, uint32, int32, int16, float16, bfloat16 |                    | ND         | 算子输入特征说明：<br>输入均为Shape为[1]的 |          |
