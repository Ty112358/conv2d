import numpy as np

#手写了卷积与池化，用于后面的图像分析

def conv2d(x,kernel,filling=False):
    '''
    x: 输入矩阵\n
    kernel: 卷积核（常用3*3、5*5）,为了能够实现填充，卷积核shape必须是奇数\n
    filling: 选择是否边缘填充，填充后边缘补零，输出大小不变
    '''
    # 读取输入和卷积核的shape
    hx,wx = x.shape[:2]
    hk,wk = kernel.shape[:2]

    # 若执行填充，对输入矩阵边缘补零，补零的宽度为 （kernelweight -1） *2
    if filling:
        filling_weight= int(0.5 * (hk - 1))
        x = np.concatenate((np.zeros((hx,filling_weight),dtype=np.float),x,np.zeros((hx,filling_weight),dtype=np.float)),axis=1)
        x = np.concatenate((np.zeros((filling_weight,wx+2*filling_weight),
        dtype=np.float),x,np.zeros((filling_weight,wx+2*filling_weight),dtype=np.float)),axis=0)
        hx,wx = x.shape[:2]
    else:
        pass

    # 计算新的行列数，新建零矩阵
    new_h,new_w = hx - hk + 1, wx - wk + 1
    output = np.zeros((new_h,new_w),dtype=np.float64)

    # 运算
    for h in range(new_h):
        for w in range(new_w):
            output[h,w] = np.sum(x[h:h+hk,w:w+wk] * kernel)

    return output


if __name__ == "__main__":
    print(conv2d(np.ones((5,5),np.float),np.ones((3,3),np.float),True))
