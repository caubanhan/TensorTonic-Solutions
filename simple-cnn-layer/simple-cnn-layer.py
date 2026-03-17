import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    # H, W_in = x.shape
    # fH, fW = W.shape
    
    # # 2. Tính kích thước đầu ra (Formula: N - F + 1)
    # out_h = H - fH + 1
    # out_w = W_in - fW + 1
    
    # # 3. Khởi tạo mảng kết quả với giá trị 0
    # Y = np.zeros((out_h, out_w))
    
    # # 4. Vòng lặp trượt Filter trên ảnh
    # for h in range(out_h):        # Duyệt theo chiều cao đầu ra
    #     for w in range(out_w):    # Duyệt theo chiều rộng đầu ra
            
    #         # Trích xuất vùng cửa sổ (vùng ảnh con mà Filter đè lên)
    #         # x[h : h + fH, w : w + fW]
            
    #         # Tính tích chập tại vị trí (h, w)
    #         sum_val = 0
    #         for i in range(fH):
    #             for j in range(fW):
    #                 sum_val += x[h + i, w + j] * W[i, j]
            
    #         # Cộng thêm bias và gán vào kết quả
    #         Y[h, w] = sum_val + b
            
    # return Y

    """
    x: (Batch, Channels, Height, Width)
    W: (Num_Filters, Channels, K_Height, K_Width)
    b: (Num_Filters,)
    """

    """ CÁCH 2
    # 1. Lấy kích thước
    N, C, Xh, Xw = x.shape
    F, C_W, Kh, Kw = W.shape
    
    # Kiểm tra xem số lượng Channel của ảnh và Filter có khớp nhau không
    assert C == C_W, "Số lượng channels của x và W phải bằng nhau"
    
    # 2. Tính kích thước đầu ra
    out_h = Xh - Kh + 1
    out_w = Xw - Kw + 1
    
    # 3. Khởi tạo Y: (Batch, Num_Filters, Out_H, Out_W)
    Y = np.zeros((N, F, out_h, out_w))
    
    # 4. Vòng lặp
    for n in range(N):             # Lặp qua từng ảnh trong Batch
        for f in range(F):         # Lặp qua từng Filter
            for h in range(out_h):
                for w in range(out_w):
                    # Slicing trên 4 chiều:
                    # [n, :, h:h+Kh, w:w+Kw] -> Lấy ảnh thứ n, tất cả các kênh, vùng (h, w)
                    # W[f] -> Lấy filter thứ f (có đủ các kênh tương ứng)
                    window = x[n, :, h:h+Kh, w:w+Kw]
                    
                    # Nhân chập: Tổng của (tất cả các kênh * filter tương ứng)
                    Y[n, f, h, w] = np.sum(window * W[f]) + b[f]
                    
    return Y
    """
    
    N, C, Xh, Xw = x.shape
    F, C_W, Kh, Kw = W.shape
    
    out_h = Xh - Kh + 1
    out_w = Xw - Kw + 1
    Y = np.zeros((N, F, out_h, out_w))

    for h in range(out_h):
        for w in range(out_w):
            window = x[:,:, h:h+Kh, w:w+Kw]
            res = np.sum(window[:, np.newaxis, :, :, :] * W[np.newaxis, :, :, :, :], axis=(2, 3, 4))
            
            # 3. Cộng bias b (C_out,) và gán vào output
            Y[:, :, h, w] = res + b
    return Y