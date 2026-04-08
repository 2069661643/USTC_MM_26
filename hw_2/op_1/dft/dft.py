"""
Code by UTF-8
"""

from PIL import Image
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import numpy as np
import math
import os
import struct

def GetFig(filepath,show=True):
    """
    读取图片,返回tuple[4]
    1.image.size : tuple[2]
    2.ColorRed矩阵 : np.array
    3.ColorGreen矩阵 : np.array
    4.ColorBlue矩阵 : np.array
    """
    image = Image.open(filepath)
    if show:
        image.show()
    width, height = image.size
    ColorRed=np.zeros((height,width),dtype=complex)
    ColorGreen=np.zeros((height,width),dtype=complex)
    ColorBlue=np.zeros((height,width),dtype=complex)
    for i in range(height):
        for j in range(width):
            pixel=image.getpixel((j,i))
            ColorRed[i,j]=pixel[0]
            ColorGreen[i,j]=pixel[1]
            ColorBlue[i,j]=pixel[2]
    return (image.size,ColorRed,ColorGreen,ColorBlue)
    
def DFTMatCompress(Mat:np.array,size:int|str=8,cut:bool=True,show:bool=True):
    """
    DFT变换
    变量
    Mat: 待DFT处理矩阵
    size: 剪切单位区间，默认为8
    cut: 是否减去边缘
    show: 展示输出部分矩阵
    返回DFT矩阵
    """
    #确认DFT矩阵大小
    if size == "max":
        m,n=Mat.shape
    else :
        m=size
        n=size
    #确定大小
    s,t=Mat.shape
    row=s//m
    line=t//n
    #确定函数
    def funcm(x,y):
        return np.exp(-2*complex(0,1)*np.pi*(x*y/m))
    def funcn(x,y):
        return np.exp(-2*complex(0,1)*np.pi*(x*y/n))
    #构造DFT矩阵
    DFTMatM=np.fromfunction(funcm,(m,m))
    DFTMatN=np.fromfunction(funcn,(n,n))
    #确定剪切
    if cut:
        FourierMat=np.zeros((row*m,line*n),dtype=complex)
    else:
        if s%m!=0 or t%m!=0:
            raise ValueError("无法整除！")
        FourierMat=np.zeros((s,t),dtype=complex)
    #处理主过程
    for p in range(row):
        for q in range(line):
            TempMat=DFTMatM.dot(Mat[p*m:(p+1)*m,q*n:(q+1)*n].dot(DFTMatN))
            FourierMat[p*m:(p+1)*m,q*n:(q+1)*n]=TempMat
    #展示
    if show:
        print('以下为DFT后部分矩阵:')
        if s>8 or t>8:
            print(FourierMat[:8,:8]/(np.sqrt(m*n)))
        else :
            print(FourierMat/(np.sqrt(m*n)))
    #输出
    return FourierMat/(np.sqrt(m*n))

def DDFTMatCompress(Mat:np.array,size:int|str=8,cut:bool=True,show:bool=True):
    """
    DDFT变换
    变量
    Mat: 待DDFT处理矩阵
    size: 剪切单位区间，默认为8
    cut: 是否减去边缘
    show: 展示输出部分矩阵
    返回
    DDFT矩阵
    """
    #确认DFT矩阵大小
    if size == "max":
        m,n=Mat.shape
    else :
        m=size
        n=size
    #确定大小
    s,t=Mat.shape
    row=s//m
    line=t//n
    #确定函数
    def funcm(x,y):
        return np.exp(2*complex(0,1)*np.pi*(x*y/m))
    def funcn(x,y):
        return np.exp(2*complex(0,1)*np.pi*(x*y/n))
    #构造DFT矩阵
    DFTMatM=np.fromfunction(funcm,(m,m))
    DFTMatN=np.fromfunction(funcn,(n,n))
    #确定剪切
    if cut:
        FourierMat=np.zeros((row*m,line*n),dtype=complex)
    else:
        if s%m!=0 or t%m!=0:
            raise ValueError("无法整除！")
        FourierMat=np.zeros((s,t),dtype=complex)
    #处理主过程
    for p in range(row):
        for q in range(line):
            TempMat=DFTMatM.dot(Mat[p*m:(p+1)*m,q*n:(q+1)*n].dot(DFTMatN))
            FourierMat[p*m:(p+1)*m,q*n:(q+1)*n]=TempMat
    #展示
    if show:
        print('以下为DDFT后部分矩阵:')
        if m>8 or n>8:
            print(FourierMat[:8,:8]/(np.sqrt(m*n)))
        else :
            print(FourierMat/(np.sqrt(m*n)))
    #输出
    return (FourierMat)/(np.sqrt(m*n))

def ShowFig(Rmat,Gmat=None,Bmat=None,show=True,save:bool|str=False):
    """
    变量
    Rmat: R颜色矩阵
    其余二者同理
    后两者缺省视作全为255
    """
    height,width=Rmat.shape
    if Gmat is None:
        Gmat=np.ones((height,width))*255
    if Bmat is None:
        Bmat=np.ones((height,width))*255
    #取实部
    Rm=Rmat.real
    Gm=Gmat.real
    Bm=Bmat.real

    output_image=Image.new("RGB",(width,height))
    for i in range(height):
        for j in range(width):
            output_image.putpixel((j,i),(round(Rm[i,j]),round(Gm[i,j]),round(Bm[i,j])))
    if show:
        output_image.show()
    if not (save is False):
        output_image.save(save+".bmp")
        output_image.save(save+".jpg")
    return

def ShowGrey(Mat,err:float|str=0.5,cerr=50):
    """
    显示矩阵实部绝对值灰度图
    红点为超过127
    灰点为(err,127]
    白点为[0,err]
    灰点对应实部绝对值越大，越灰，否则越白(仿射变换)
    g=255-round(l*(255-cerr)/(127-cerr)+cerr)
    变量：
    Mat: 待显示矩阵
    err: 允许误差值，即模长的允许误差，若为"auto"，自动取为矩阵所有实部平均值
    cerr: 仿射变换截距值，0~255
    """
    height,width=Mat.shape
    output_image=Image.new("RGB",(width,height))
    #参考数据显示
    aver=np.mean(Mat.real)
    sup=(Mat.real).max()
    print("aver:%g"%(aver))
    print("sup:%g"%(sup))
    if err=="auto":
        err=aver/10
    print("err=%g"%(err))
    #转化灰度图数据
    for i in range(height):
        for j in range(width):
            if abs(Mat.real[i,j])<=err:
                #绘制白点
                r,g,b=255,255,255
            elif abs(Mat.real[i,j])>=127:
                #绘制红点
                r,g,b=255,0,0
            else:
                #仿射变换
                l=abs(Mat.real[i,j])
                g=255-round(l*(255-cerr)/(127-cerr)+cerr)
                r,b=g,g
            output_image.putpixel((j,i),(r,g,b))
    #显示灰度图
    output_image.show()

def ZipMat(Mat,err=50):
    """
    压缩图片，round函数取整
    并依据err将模长在[0,err]的值抹为0
    变量：
    Mat: 待显示矩阵
    err: 允许误差值，即模长的允许误差，若为"auto"，自动取为矩阵所有实部平均值
    cerr: 颜色显示跃进值，0~255
    返回：
    处理后矩阵np.array
    """
    height,width=Mat.shape
    outputmat=np.zeros((height,width),dtype=complex)
    count=0
    #处理数据
    for i in range(height):
        for j in range(width):
            rm=Mat.real[i,j]
            im=Mat.imag[i,j]
            if abs(Mat[i,j])<=err:
                #抹0
                count+=1
                continue
            outputmat[i,j]=complex(round(rm),round(im))
    #计算0与矩阵大小比
    print("理论最大压缩比:%g%%"%(100-count*100/height/width))
    return outputmat

def ZipFig(RMat,GMat,BMat,filepath="bifig.bin"):
    """
    将RGB三矩阵放入二进制文件
    输入:
    RMat,GMat,BMat: 复数np.array
    filepath: 储存文件地址
    """
    binfile = open(filepath, 'wb')
    m,n=RMat.shape
    #二进制开头
    bm=struct.pack("i",m)
    bn=struct.pack("i",n)
    binfile.write(bm)
    binfile.write(bn)
    #规定特殊数值
    boverflow=struct.pack("b",-128)
    zero=struct.pack("b",0)
    for Mat in [RMat.real,GMat.real,BMat.real,RMat.imag,GMat.imag,BMat.imag]:
        countof0=0
        for i in range(m):
            for j in range(n):
                data=Mat[i,j]
                if round(data)!=0 and countof0!=0:
                    #持续为0后不为零，将前者打包
                    bcount=struct.pack("i",countof0)
                    countof0=0
                    binfile.write(zero)
                    binfile.write(bcount)
                if abs(data)>=127:
                    #超过short范围的数据处理
                    binfile.write(boverflow)
                    bdata=struct.pack("f",data)
                    binfile.write(bdata)
                elif round(data)==0:
                    #为0
                    countof0+=1
                else:
                    #写入正常数据
                    bdata=struct.pack("b",round(data))
                    binfile.write(bdata)
        #尾处理
        if countof0!=0:
            bcount=struct.pack("i",countof0)
            countof0=0
            binfile.write(zero)
            binfile.write(bcount)
    binfile.close()

def DezipFig(filepath="bifig.bin"):
    """
    从二进制文件中提取RGB三矩阵
    输入:
    filepath: 文件地址
    输出(tuple[3]):
    RGB三矩阵，复数np.array
    """
    binfile = open(filepath, 'rb')
    try:
        #读取矩阵大小
        bm=binfile.read(4)
        bn=binfile.read(4)
        m=struct.unpack("i",bm)[0]
        n=struct.unpack("i",bn)[0]
        RMatR=np.zeros((m,n))
        GMatR=np.zeros((m,n))
        BMatR=np.zeros((m,n))
        RMatI=np.zeros((m,n))
        GMatI=np.zeros((m,n))
        BMatI=np.zeros((m,n))
        for Mat in [RMatR,GMatR,BMatR,RMatI,GMatI,BMatI]:
            countof0=0
            for i in range(m):
                for j in range(n):
                    if countof0 == 0:
                        #前非0
                        bdata=binfile.read(1)
                        data=struct.unpack("b",bdata)[0]
                        if data==-128:
                            #超short数据
                            bf=binfile.read(4)
                            f=struct.unpack("f",bf)[0]
                            Mat[i,j]=f
                        elif data==0:
                            #为0
                            bzero=binfile.read(4)
                            countof0=struct.unpack("i",bzero)[0]-1
                            Mat[i,j]=0
                        else :
                            #正常数据
                            Mat[i,j]=data
                    else:
                        #前为0
                        countof0-=1
                        Mat[i,j]=0
    finally:
        binfile.close()
    return RMatR+RMatI*complex(0,1),GMatR+GMatI*complex(0,1),BMatR+BMatI*complex(0,1)

def Test2():
    import random
    random.seed(123)
    Mat=np.arange(36)
    for i in range(36):
        Mat[i]=random.randint(0,36)
    Mat.resize((6,6))
    """for i in range(6):
        for j in range(6):
            mr=Mat[i,j]
            print("%d&"%(mr),end="")
        print("\\\\")"""

    DFTMat=DFTMatCompress(Mat,"max")
    """Mr=DFTMat.real
    Mi=DFTMat.imag
    for mat in [Mr,Mi]:
        for i in range(6):
            for j in range(6):
                x=round(mat[i,j])
                if j==5 :
                    print("%d"%(x),end="")
                    continue
                print("%d&"%(x),end="")
            print("\\\\")
        print("")"""

def Test1():
    Mat=np.arange(36)
    Mat.resize((6,6))
    '''for i in range(6):
        for j in range(6):
            mr=Mat[i,j]
            print("%d&"%(mr),end="")
        print("\\\\")
    '''
    DFTMat=DFTMatCompress(Mat,"max")
    '''Mr=DFTMat.real
    Mi=DFTMat.imag
    for mat in [Mr,Mi]:
        for i in range(6):
            for j in range(6):
                x=round(mat[i,j])
                if j==5 :
                    print("%d"%(x),end="")
                    continue
                print("%d&"%(x),end="")
            print("\\\\")
        print("")'''
    
def Test3():
    T,R,G,B=GetFig("Example.jpg")
    dr=DFTMatCompress(R,size=16,show=False)
    ShowGrey(dr,20)
    dr=DFTMatCompress(R,size="max",show=False)
    ShowGrey(dr,20)

def Test(name):
    import time
    ds=16
    T,R,G,B=GetFig("%s.jpg"%(name))
    print("16xDFT for %s"%(name))
    dr=DFTMatCompress(R,size=ds,show=False)
    zr=ZipMat(dr,50)
    dg=DFTMatCompress(G,size=ds,show=False)
    zg=ZipMat(dg,50)
    db=DFTMatCompress(B,size=ds,show=False)
    zb=ZipMat(db,50)
    ZipFig(zr,zg,zb,"16x%s.bin"%(name))
    start=time.time()
    zr,zg,zb=DezipFig("16x%s.bin"%(name))
    r=DDFTMatCompress(zr,size=ds,show=False)
    g=DDFTMatCompress(zg,size=ds,show=False)
    b=DDFTMatCompress(zb,size=ds,show=False)
    ShowFig(r,g,b,show=False,save="16x%s"%(name))
    end=time.time()
    print("用时%.4gs"%(-start+end))
    ds="max"
    print("maxDFT for %s"%(name))
    dr=DFTMatCompress(R,size=ds,show=False)
    zr=ZipMat(dr,50)
    dg=DFTMatCompress(G,size=ds,show=False)
    zg=ZipMat(dg,50)
    db=DFTMatCompress(B,size=ds,show=False)
    zb=ZipMat(db,50)
    ZipFig(zr,zg,zb,"max%s.bin"%(name))
    start=time.time()
    r,g,b=DezipFig("max%s.bin"%(name))
    r=DDFTMatCompress(zr,size=ds,show=False)
    g=DDFTMatCompress(zg,size=ds,show=False)
    b=DDFTMatCompress(zb,size=ds,show=False)
    ShowFig(r,g,b,show=False,save="max%s"%(name))
    end=time.time()
    print("用时%.4gs"%(-start+end))

if __name__=="__main__":
    for i in [1,2,3]:
        Test("Example%d"%(i))
