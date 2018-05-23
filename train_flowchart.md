graph TD;
   A[输入图像]-->B[基于ResNet的U-Net结构];
   A-->G[计算ground truth]
   B-->C[使用卷积得到预测的pixel-based的像素信息]
   B-->D[使用卷积得到预测的集合信息]
   G-->E
   C-->E[和ground truth计算balanced cross-entropy loss]
   G-->F
   D-->F[和ground truth计算geometry loss]