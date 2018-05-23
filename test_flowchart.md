graph TD;
   A[输入图像]-->B[基于ResNet的U-Net结构];
   B-->C[使用卷积得到预测的pixel-based的像素信息]
   B-->D[使用卷积得到预测的几何信息]
   C-->E["挑选出大于阈值(比如说0.8)的pixel"]
   D-->F   
   E-->F["选择满足阈值条件的pixel的score和几何信息"]
   F-->G[针对每个pixel计算得到一个bounding box]
   G-->H[针对上述得到的bounding box进行非最大抑制]
   H-->I[得到text的bounding box]