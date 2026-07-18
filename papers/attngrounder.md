#### AttnGrounder: Talking to Cars with Attention
Vivek Mittal
Indian Institute of Technology, Mandi, India
b18153@students.iitmandi.ac.in
```
Abstract. We propose Attention Grounder (AttnGrounder), a single-
```
stage end-to-end trainable model for the task of visual grounding. Visual
grounding aims to localize a specific object in an image based on a given
natural language text query. Unlike previous methods that use the same
text representation for every image region, we use a visual-text attention
module that relates each word in the given query with every region in
the corresponding image for constructing a region dependent text rep-
resentation. Furthermore, for improving the localization ability of our
model, we use our visual-text attention module to generate an atten-
tion mask around the referred object. The attention mask is trained as
an auxiliary task using a rectangular mask generated with the provided
ground-truth coordinates. We evaluate AttnGrounder on the Talk2Car
dataset and show an improvement of 3.26% over the existing methods.
Code is available at https://github.com/i-m-vivek/AttnGrounder.
```
Keywords: Object Detection, Visual Grounding, Attention Mechanism
```
1 Introduction
In recent years, there have been many advances in tasks involving the joint
processing of images and text. Researchers are working on various challenging
problems like image captioning [1, 7, 28], visual question answering [1, 11, 30],
text-conditioned image generation [13,31], etc. In this work, we address the task
of visual grounding [5, 24] in which our goal is to train a model that can localize
an image region based on a given natural language text query. The task of visual
grounding can be useful in many practical applications. In Figure 1, we provide
an example, in a self-driving car, the passenger can give a command to the car
like “Do you see that lady walking on the sidewalk, up here on the left. She
is the one we need to pick up. Pull over next to her”. Additional use cases
can be found in embodied agents and human-computer interaction. The task of
visual grounding is quite challenging as it requires joint reasoning over text and
images. Consider the example shown in Figure 1: in order to correctly identify
the women, a model first needs to understand the command properly and then
locate the target object in the image.
In this paper, we propose a novel architecture for visual grounding, namely,
the AttnGrounder. At a high-level, visual grounding consists of two sub-tasks:
object detection [21,23] and ranking detected objects. In the first task, the model
```
arXiv:2009.05684v2 [cs.CV] 11 Dec 2020
```
2 V. Mittal
Fig. 1. Example of visual grounding task. Green box indicates the referred object.
identifies all the objects present in the image, and then it calculates a match-
ing score between every object and given text query to rank different proposals.
These two tasks are themselves quite challenging to solve, and the object de-
tection task may become a bottleneck for the performance of the whole system.
In contrast to the traditional two-stage [5, 29] approaches our AttnGrounder di-
rectly operates on raw RGB images and text expressions. Our main goal is to
combine the image and text features, and then directly predict bounding boxes
from them. To this end, we use a YOLOv3 [22] backbone to generate visual fea-
```
tures (Sec. 3.1) and a Bi-LSTM to generates text features (Sec. 3.2). For jointly
```
reasoning over visual and text features, we use two sub-modules.: Text Feature
```
Matrix (Sec. 3.3.1) and Attention Map (Sec. 3.3.2). The text feature matrix cal-
```
culates text representation for every image region by selecting important words
for that particular region. Predicting a rough segmentation mask around the
referred object can also help in predicting a bounding box. Thus, we make a
rectangular mask around the target object using the ground truth coordinates,
and the goal of our attention map module is to predict that mask. The attention
map module is trained jointly with rest of the system and yields an auxiliary loss
which can improve the localization ability of our model. Finally, we fuse all the
features to obtain a multi-modal feature representation with which we predict
```
the bounding box (Sec. 3.4). In Section 2, we give an overview of the related
```
work. In Section 3, we provide a detailed description of our approach. In Section
4, we evaluate our model on the Talk2Car dataset [6]. Finally, we conclude our
paper in Section 5.
2 Related Work
In terms of the object detection subtask in visual grounding, there exist two lines
of work in visual grounding: a one-stage approach [3, 19, 24, 27] and a two-stage
approach [5, 17, 25, 26]. In most of the previous work, researchers address the
task of visual grounding with two-stage approaches. In the first stage, several
object proposals are generated using an off-the-shelf object detection algorithm
like Faster-RCNN [23], YOLO [21, 22]. In the second stage, the object proposals
are ranked by calculating the matching score with the given text query.
AttnGrounder 3
Several works [25, 26] explore graph convolutional networks for learning the
relationship between objects and text query. The work of [26] focuses on im-
proving the cross-modal relationship between objects and referring expressions
by constructing a language-guided visual relation graph. They use a gated graph
convolutional network to fuse information from different modalities. In [25], au-
thors also use graph networks for learning the relationship between neighboring
objects. NMTree [16] constructs a language parsing tree with which they localize
referred object by accumulating grounding confidence per node. MAttNet [29]
involves a two-stage modular approach in which they use three modules for
processing visual, language, location information, and a relationship module to
understand the relationship between the output of these three modules. In [18],
the authors use attention to generate difficult examples by discarding the most
important information from both text and image. A-ATT [4] constructs three
modules to understand the image, objects, and query. They accumulate features
from different modalities in a circular manner to guide the reasoning process in
multiple steps. MSRR [5] also uses a multi-step reasoning procedure, in which a
new matching score is calculated between every object and the language query
in each step. After all the reasoning steps, the object that has the highest score
is selected as the target object.
In two-stage methods, the offline object detector may become a bottleneck
as it may fail to provide good object proposals. For addressing this issue, sev-
eral recent works explore a one-stage paradigm for visual grounding. One-stage
methods fuse image-text features and then directly predict the bounding box
for the referred object. Recent works [24, 27] fuse visual-text features at every
spatial location and predict adjustment in predefined anchor boxes to align the
bounding boxes with the referred object. In [3], the authors introduce a guided
attention module with which their model also predicts the center coordinates
of the referred object along with the referred region. In our AttnGrounder, we
predict a mask rather than the center coordinate which can help in better local-
ization of image regions. Moreover, previous one-stage methods consider a single
text representation for every location, but as we know, different regions corre-
spond to different words, which is why we develop a text feature matrix where
every location has its unique text representation. Additionally, one-stage meth-
ods are faster than two-stage methods as they don’t involve matching between
various image regions and text queries.
3 Methodology
In Figure 2, we provide an overview of our proposed visual grounding architec-
ture. Our goal is to locate an object in an Image I based on a given text query T .
Our proposed model AttnGrounder consists of five sub-modules: image encoder,
text encoder, visual attention module, fusion module, and a grounding module.
The image encoder uses Darknet-53 [22] with pyramid structure [14] to extract
```
visual features from the image I in the form of grids {Gk}2k=0 ∈ RCk ×Hk ×Wk at
```
three different spatial resolutions. To encode text features, we use a Bi-LSTM
4 V. Mittal
Fig. 2. Overview of AttnGrounder.
that can easily capture the long-range dependency in given query. We introduce
a visual-text attention mechanism that attends to words wt in the text query T
at every spatial location in the grid Gk. This attention mechanism also generates
an attention map M for the referred object. Finally, we fuse the image and text
feature using 1 × 1 convolution layers. For training, we use two loss functions:
binary cross-entropy loss for training the attention map and YOLO’s loss [21]
```
function for complete end-to-end training. For our use, we modify the YOLO’s
```
loss function by replacing the last sigmiod unit with a softmax unit.
3.1 Image Encoder
We adopted Darknet-53 [22] with a pyramid network [14] structure for encod-
ing visual features. Darknet-53 takes an image I and produces feature grids
```
{Gk}2k=0 ∈ RCk ×Hk ×Wk at three different spatial resolutions, where Ck, Hk and
```
Wk are the number of channels, height and width of the grid at kth resolution.
Usually, referring expressions also contain position information about the re-
```
ferred object (eg. ”park in front of the second vehicle on our right side.”). Visual
```
features produced by Darknet lack such location information. Similar to [27], we
explicitly add location information by concatenating a vector Cij ∈ R8 at every
spatial location of the grid Gk,
```
Cij =
```
```
( i
```
Wk,
j
Hk,
i + 0.5
Wk,
j + 0.5
Hk,
i + 1
Wk,
j + 1
Hk,
1
Wk,
1
Hk
```
)
```
where i, j are row and column in the grid Gk respectively. Then, we use a 1 × 1
convolution layer with batch normalization and ReLU unit to map these grid
feature to a common semantic dimension D. Now, Gk ∈ RCk ×Hk ×D .
3.2 Text Encoder
Our text encoder consists of an embedding layer and a Bi-LSTM layer. The text
```
query T of length n is first converted to its embedding Q = {ei}n−1i=0 using the
```
embedding layer. After that, Q is fed as an input to Bi-LSTM that generates
two hidden states [hrighti , hlef ti ] for every word embedding ei. The forward and
AttnGrounder 5
backward hidden states are concatenated to get hi ∈ R2L, where L is the number
of hidden units in Bi-LSTM. Furthermore, we use a linear layer to project the
text embedding into the common semantic space of dimension D. Embeddings
for all n words are stacked to get Q′ ∈ Rn×D .
3.3 Visual-Text Attention
Fig. 3. Visual-Text Attention Module
Every region in an image may correspond to different words in the given
query. To better model the dependency between a word and a region, we present
a visual-text attention module. Fig. 3 illustrates this module. This module gets a
grid Gk from the image encoder and text embeddings Q′ from the text encoder.
The grid Gk ∈ RHk ×Wk ×D has Hk × Wk spatial locations each of dimension D
and text embeddings Q′ ∈ Rn×D have n word features each of dimension D. To
measure the matching score between a location gi ∈ RD and word wj ∈ RD in
the query, we calculate the dot product between them. We multiply Gk with Q′,
```
which generates a matrix M k = Gk · (Q′)T , where M kij represents the matching
```
```
score between grid location i and word j. The matrix M k ∈ R(Hk ×Wk )×n, where
```
Hk and Wk are the height and width of the grid at kth resolution generated by
our image encoder. This matrix M k is used in two ways described next.
3.3.1 Text Feature Matrix For every location in the grid Gk, we can repre-
sent it’s text features by summing up word features from the matrix Q′ ∈ Rn×D .
We normalize the matrix M k using the softmax function to generate a word-level
6 V. Mittal
```
attention matrix αk ∈ R(Hk ×Wk )×n, where αkij denotes the correlation between
```
region i in the grid Gk and word j in the text query T .
```
αkij = exp(M
```
```
kij )
```
∑n−1
```
j=0 exp(M kij )
```
```
(1)
```
To obtain text features at every location, we multiply αk and Q′, which
```
generates text features matrix T ′k ∈ R(Hk ×Wk )×D , denoted as T ′k = αk ·Q′. Every
```
row in the matrix T ′k represents text features, which are effectively derived from
the word features weighted by the correlation between words and corresponding
image region. Thus, every region has its unique text representation.
3.3.2 Attention Map In visual grounding, a rectangular mask around the
referred object can be predicted by just using the provided ground truth bound-
ing boxes. Our attention map module aims to make a rectangular mask around
the referred object, which is an auxiliary task that helps improve the overall
performance. By learning to make a mask around the object, the model can
indirectly learn to locate objects based on the given query and image. We make
```
use of the matrix M k ∈ R(Hk ×Wk )×n, which already contains the matching score
```
between every image region and every word in the text query. We sum the values
```
in every row of M k to get a column matrix βk ∈ R(Hk ×Wk )
```
βki =
n−1∑
```
j=0
```
```
M kij (2)
```
βki provides the aggregated matching score for an image region i and the given
text query T . Now, the image regions matching with the given query will have
a high matching score and vice-versa. To create a mask, we feed βk through a
sigmoid unit, which maps the values of βk between 0 and 1. Therefore, βki can
be interpreted as the probability of having the referred object in the region i.
For training, we generate a rectangular mask using the ground truth coordi-
nates. The ground truth mask has the value of 1 inside the rectangular region
and 0 elsewhere. Masks are created for all three resolutions and training is done
```
using binary cross-entropy loss (See Figure 4). At the kth resolution, given the
```
```
ground truth mask βktrue ∈ RHk ×Wk and the predicted mask βk ∈ RHk ×Wk ; the
```
mask loss is calculated as
```
Lkmask = − 1H
```
k × Wk
```
(Hk ×Wk )−1∑
```
```
i=0
```
```
(βki,true log(βki ) + (1 − βki,true) log(1 − βki )) (3)
```
The Lmask is obtained by summing up the Lkmask terms calculated for all the
three resolutions.
```
Lmask =
```
2∑
```
k=0
```
```
Lkmask (4)
```
AttnGrounder 7
Fig. 4. The mask loss is calculated for every spatial resolution using binary cross-
entropy loss and summed to get the final mask loss. First row shows attention map
produced by our AttnGrounder at three different resolutions and second row shows
ground truth mask used for training.
Using Lmask as an auxiliary loss can help in embedding visual and text features
in the common semantic space D, by encouraging similar visual and text features
to be nearby and vice-versa. Furthermore, it can also help discriminate between
the referred region and the others.
We also generate attended visual features Vk ∈ RHk ×Wk ×D by an element-
```
wise product between the attention map βk and grid Gk, Vk = βk Gk (
```
```
denotes element-wise product). In Vk, every spatial location is scaled by its
```
importance determined by the attention mask βk. Thus, visual features of regions
where target object may be present are enhanced and visual features of other
redundant regions are reduced.
3.4 Fusion Module
For fusing image and text features, we concatenate visual feature grid Gk, text
feature matrix T ′k and attended visual features Vk along the channel dimension.
This generates a matrix Fk ∈ RHk ×Wk ×3D . All the features are l2 normalized
before concatenation and fusion is done for all the three spatial resolutions gen-
erated by our image encoder. After that, we use a 1 × 1 convolution layer to fuse
these visual and text features. This 1 × 1 convolution layer maps these fused
features in a new semantic space of dimension D′. After fusion, we have three
```
feature matrices {Fk}2k=0 ∈ RHk ×Wk ×D′.
```
3.5 Grounding Module
The grounding module aims to ground the text query onto an image region. This
module takes the fused feature vector Fk as input and generates a bounding box
prediction. We follow [27] for designing this module and similar to [27] we replace
YOLO’s [21] output sigmoid layer with softmax layer.
8 V. Mittal
Fig. 5. Examples of results obtained using our AttnGrounder. The predicted bounding
boxes are shown in red and ground truth bounding boxes are shown in green. The
second row shows the attention maps generated by our model.
For every spatial location, YOLOv3 centres three different anchor boxes and
for every spatial resolution, YOLOv3 uses different set of anchor boxes. In total,
for three different spatial resolutions we have 3×3 = 9 different predefined anchor
boxes. We denote the total number of anchor box predictions made by our model
with m, where m = ∑2k=0 Hk × Wk × 3. For every anchor box, YOLOv3 predicts
changes required in location and size to fit that particular anchor box around the
target object. Specifically, YOLOv3 uses two branches: the first branch predicts
the shift in centre, height, width of the predefined anchor box and the second
```
one uses a sigmoid layer to predict the confidence (on the shifted box) of being
```
the target box. As we need only one bounding box prediction for grounding, we
replace the last sigmoid layer with softmax layer which forces to select one box
out of the m boxes as prediction. Accordingly, the loss function for confidence is
```
also changed to a cross-entropy loss between the predicted confidence (softmax
```
```
version) and a one-hot vector with a 1 entry corresponding to the anchor box
```
that has the highest intersection over union with the ground truth box. We
refer the readers to [21, 22] for more details and abstract away the YOLO’s loss
```
function as Lyolo. Thus, total loss becomes
```
```
L = Lyolo + λLmask (5)
```
where λ is a hyper-parameter which scales the loss Lmask.
4 Experiments and Results
In this section, we provide details of our experiments. We evaluate our approach
on Talk2Car dataset [6] and compare with five baselines.
4.1 Dataset
The Talk2Car dataset is based on nuScenes dataset [2] and contains 11,959
referring expressions for 9,217 images. This dataset contains images of a city-
AttnGrounder 9
Table 1. Comparison with state-of-the-art methods on Talk2Car testset
```
Method AP50 Score Time (ms) Params (M)
```
STACK [8] 33.71 52 35.2
SCRC [9] 43.80 208 52.47
A-ATT [4] 45.12 180 160.31
MAC [10] 50.51 51 41.59
MSRR [5] 60.04 270.5 62.25
AttnGrounder 63.30 25.5 75.84
like environment where a self-driving car is driving. Talk2Car dataset is a multi-
modal dataset that consists of various sensor modalities such as semantic maps,
LIDAR, 360-degree RGB images, etc. However, we only use the RGB images
for training our AttnGrounder. RGB images contained in Talk2Car dataset are
```
taken in different weather conditions (sunny, rainy) and time of day (day, night)
```
which makes it even more challenging. Additional challenges that the Talk2Car
dataset presents are the ambiguity between the objects belonging to the same
class, far away object localization, long text queries, etc. In this dataset, the
size of images is 900 × 1600 and on average, the referring expressions contain 11
words. The dataset contains 8349, 1163 and 2447 images for training, validation
and testing respectively. Figure 1 shows an example of Talk2Car dataset.
4.2 Training Details
Our image encoder backbone i.e. Darknet-53 [22] is pre-trained on COCO [15]
object objection task. We use 300 dimensional GloVe [20] embeddings for initial-
izing word vectors. Words for which the GloVe embeddings were not available
were initialized randomly. We resize the images to 416 × 416 and preserve the
original aspect ratio. We resize the longer edge to 416 and pad the shorter edge
with the mean pixel value. We also add some data augmentations i.e., horizontal
flips, changing saturation and intensity, random affine transformation. The an-
```
chors used in our grounding module are (10,13), (16,30), (33,23), (30,61), (62,45),
```
```
(59,119), (116,90), (156,198), (373,326). We use λ = 0.1 (defined in Sec. 3.5) for
```
training. We train our model using Adam [12] optimizer with an initial learning
rate of 10−4 and polynomial learning rate scheduler is used with a power of 1.
For the pre-trained Darknet-53 portion, we keep a lower initial learning rate of
10−3. We keep a batch size of 14 for training our AttnGrounder.
4.3 Comparison Metrics
We evaluate our approach on three different metrics. First one is AP50 score,
which is defined as the percentage of predicted bounding boxes that have an
Intersection Over Union of more than 0.5 with the ground truth bounding boxes.
The second one is inference time and third is number of parameters in a method.
10 V. Mittal
4.4 Results
We compare our AttnGrounder with four baselines and state-of-the-art model
MSRR [5], results on the test set of Talk2Car dataset are shown in Table 1.
AttnGrounder outperforms all the baselines and improves state-of-the-art by
3.26% in terms of AP50 score. Thanks to our one-stage approach, we also achieve
lowest inference time among all the baseline methods. We provide examples of
prediction made by our method in Figure. 5 along with the visualization of
```
attention map generated by visual-text attention module (Sec. 3.3).
```
5 Conclusions
We propose AttnGrounder, a single-stage end-to-end trainable visual ground-
ing model. Our AttnGrounder is fast compared to two-stage approaches as it
does not involve matching various region proposals and text queries or multi-
step reasoning. We combine visual features extracted from YOLOv3 and text
features extracted from Bi-LSTM to obtain a multi-modal feature representa-
tion with which we directly predict the bounding box for the target object. Our
AttnGrounder also generates an attention map that localizes the potential spa-
tial locations where the referred object may be present. This attention map is
trained as an auxiliary task which helps improve the overall performance of our
model. Finally, we evaluate our proposed method on the Talk2car dataset and
show that it outperforms all the baseline methods.
Acknowledgements
The author wants to thank Thierry Deruyttere, Dusan Grujicic, and Monika
Jyotiyana for their feedback on an early draft of this paper. The author is grateful
to Thierry Deruyttere for his guidance and constant support.
AttnGrounder 11
References
1. Anderson, P., He, X., Buehler, C., Teney, D., Johnson, M., Gould, S., Zhang,
L.: Bottom-up and top-down attention for image captioning and visual question
answering. In: Proceedings of the IEEE conference on computer vision and pattern
```
recognition. pp. 6077–6086 (2018)
```
2. Caesar, H., Bankiti, V., Lang, A.H., Vora, S., Liong, V.E., Xu, Q., Krishnan, A.,
Pan, Y., Baldan, G., Beijbom, O.: nuscenes: A multimodal dataset for autonomous
driving. In: Proceedings of the IEEE/CVF Conference on Computer Vision and
```
Pattern Recognition. pp. 11621–11631 (2020)
```
3. Chen, X., Ma, L., Chen, J., Jie, Z., Liu, W., Luo, J.: Real-time referring expression
comprehension by single-stage grounding network. arXiv preprint arXiv:1812.03426
```
(2018)
```
4. Deng, C., Wu, Q., Wu, Q., Hu, F., Lyu, F., Tan, M.: Visual grounding via accu-
mulated attention. In: Proceedings of the IEEE conference on computer vision and
```
pattern recognition. pp. 7746–7755 (2018)
```
5. Deruyttere, T., Collell, G., Moens, M.F.: Giving commands to a self-driving car: A
```
multimodal reasoner for visual grounding. arXiv preprint arXiv:2003.08717 (2020)
```
6. Deruyttere, T., Vandenhende, S., Grujicic, D., Van Gool, L., Moens, M.F.:
```
Talk2car: Taking control of your self-driving car. arXiv preprint arXiv:1909.10838
```
```
(2019)
```
7. Feng, Y., Ma, L., Liu, W., Luo, J.: Unsupervised image captioning. In: Proceedings
of the IEEE conference on computer vision and pattern recognition. pp. 4125–4134
```
(2019)
```
8. Hu, R., Andreas, J., Darrell, T., Saenko, K.: Explainable neural computation via
stack neural module networks. In: Proceedings of the European conference on com-
```
puter vision (ECCV). pp. 53–69 (2018)
```
9. Hu, R., Xu, H., Rohrbach, M., Feng, J., Saenko, K., Darrell, T.: Natural language
object retrieval. In: Proceedings of the IEEE Conference on Computer Vision and
```
Pattern Recognition. pp. 4555–4564 (2016)
```
10. Hudson, D.A., Manning, C.D.: Compositional attention networks for machine rea-
```
soning. arXiv preprint arXiv:1803.03067 (2018)
```
11. Jiang, H., Misra, I., Rohrbach, M., Learned-Miller, E., Chen, X.: In defense of
grid features for visual question answering. In: Proceedings of the IEEE/CVF
```
Conference on Computer Vision and Pattern Recognition. pp. 10267–10276 (2020)
```
12. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint
```
arXiv:1412.6980 (2014)
```
13. Li, B., Qi, X., Lukasiewicz, T., Torr, P.: Controllable text-to-image generation. In:
```
Advances in Neural Information Processing Systems. pp. 2065–2075 (2019)
```
14. Lin, T.Y., Doll´ar, P., Girshick, R., He, K., Hariharan, B., Belongie, S.: Feature
pyramid networks for object detection. In: Proceedings of the IEEE conference on
```
computer vision and pattern recognition. pp. 2117–2125 (2017)
```
15. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Doll´ar, P.,
Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference
```
on computer vision. pp. 740–755. Springer (2014)
```
16. Liu, D., Zhang, H., Wu, F., Zha, Z.J.: Learning to assemble neural module tree net-
works for visual grounding. In: Proceedings of the IEEE International Conference
```
on Computer Vision. pp. 4673–4682 (2019)
```
17. Liu, J., Wang, L., Yang, M.H.: Referring expression generation and comprehension
via attributes. In: Proceedings of the IEEE International Conference on Computer
```
Vision. pp. 4856–4864 (2017)
```
12 V. Mittal
18. Liu, X., Wang, Z., Shao, J., Wang, X., Li, H.: Improving referring expression
grounding with cross-modal attention-guided erasing. In: Proceedings of the IEEE
```
Conference on Computer Vision and Pattern Recognition. pp. 1950–1959 (2019)
```
19. Luo, G., Zhou, Y., Sun, X., Cao, L., Wu, C., Deng, C., Ji, R.: Multi-task col-
laborative network for joint referring expression comprehension and segmentation.
```
In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
```
```
Recognition. pp. 10034–10043 (2020)
```
20. Pennington, J., Socher, R., Manning, C.D.: Glove: Global vectors for word repre-
sentation. In: Proceedings of the 2014 conference on empirical methods in natural
```
language processing (EMNLP). pp. 1532–1543 (2014)
```
21. Redmon, J., Divvala, S., Girshick, R., Farhadi, A.: You only look once: Unified,
real-time object detection. In: Proceedings of the IEEE conference on computer
```
vision and pattern recognition. pp. 779–788 (2016)
```
22. Redmon, J., Farhadi, A.: Yolov3: An incremental improvement. arXiv preprint
```
arXiv:1804.02767 (2018)
```
23. Ren, S., He, K., Girshick, R., Sun, J.: Faster r-cnn: Towards real-time object detec-
tion with region proposal networks. In: Advances in neural information processing
```
systems. pp. 91–99 (2015)
```
24. Sadhu, A., Chen, K., Nevatia, R.: Zero-shot grounding of objects from natural lan-
guage queries. In: Proceedings of the IEEE International Conference on Computer
```
Vision. pp. 4694–4703 (2019)
```
25. Wang, P., Wu, Q., Cao, J., Shen, C., Gao, L., Hengel, A.v.d.: Neighbourhood
```
watch: Referring expression comprehension via language-guided graph attention
```
networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern
```
Recognition. pp. 1960–1968 (2019)
```
26. Yang, S., Li, G., Yu, Y.: Cross-modal relationship inference for grounding referring
expressions. In: 2019 IEEE/CVF Conference on Computer Vision and Pattern
```
Recognition (CVPR). pp. 4140–4149 (2019)
```
27. Yang, Z., Gong, B., Wang, L., Huang, W., Yu, D., Luo, J.: A fast and accurate
one-stage approach to visual grounding. In: Proceedings of the IEEE International
```
Conference on Computer Vision. pp. 4683–4693 (2019)
```
28. You, Q., Jin, H., Wang, Z., Fang, C., Luo, J.: Image captioning with semantic
attention. In: Proceedings of the IEEE conference on computer vision and pattern
```
recognition. pp. 4651–4659 (2016)
```
29. Yu, L., Lin, Z., Shen, X., Yang, J., Lu, X., Bansal, M., Berg, T.L.: Mattnet: Mod-
ular attention network for referring expression comprehension. In: Proceedings of
the IEEE Conference on Computer Vision and Pattern Recognition. pp. 1307–1315
```
(2018)
```
30. Yu, Z., Yu, J., Cui, Y., Tao, D., Tian, Q.: Deep modular co-attention networks for
visual question answering. In: Proceedings of the IEEE conference on computer
```
vision and pattern recognition. pp. 6281–6290 (2019)
```
31. Zhang, H., Xu, T., Li, H., Zhang, S., Wang, X., Huang, X., Metaxas, D.N.: Stack-
```
gan: Text to photo-realistic image synthesis with stacked generative adversarial
```
networks. In: Proceedings of the IEEE international conference on computer vi-
```
sion. pp. 5907–5915 (2017)
```