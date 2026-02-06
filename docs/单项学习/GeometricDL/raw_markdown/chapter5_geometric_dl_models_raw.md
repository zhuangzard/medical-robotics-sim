# 5 Geometric Deep Learning Models

--- Page 72 ---
68
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
5
Geometric Deep Learning Models
Having thoroughly studied various instantiations of our Geometric Deep
Learning blueprint (for diﬀerent choices of domain, symmetry group, and
notions of locality), we are ready to discuss how enforcing these prescriptions
can yield some of the most popular deep learning architectures.
Our exposition, once again, will not be in strict order of generality. We ini-
tially cover three architectures for which the implementation follows nearly-
directly from our preceding discussion: convolutional neural networks
(CNNs), group-equivariant CNNs, and graph neural networks (GNNs).
We will then take a closer look into variants of GNNs for cases where a
graph structure is not known upfront (i.e. unordered sets), and through
our discussion we will describe the popular Deep Sets and Transformer
architectures as instances of GNNs.
Following our discussion on geometric graphs and meshes, we ﬁrst describe
equivariant message passing networks, which introduce explicit geometric
symmetries into GNN computations. Then, we show ways in which our
theory of geodesics and gauge symmetries can be materialised within deep
learning, recovering a family of intrinsic mesh CNNs (including Geodesic
CNNs, MoNet and gauge-equivariant mesh CNNs).
Lastly, we look back on the grid domain from a temporal angle. This discus-
sion will lead us to recurrent neural networks (RNNs). We will demonstrate
a manner in which RNNs are translation equivariant over temporal grids,
but also study their stability to time warping transformations. This property
is highly desirable for properly handling long-range dependencies, and en-
forcing class invariance to such transformations yields exactly the class of
gated RNNs (including popular RNN models such as the LSTM or GRU).
While we hope the above canvasses most of the key deep learning archi-
tectures in use at the time of writing, we are well aware that novel neural
network instances are proposed daily. Accordingly, rather than aiming to
cover every possible architecture, we hope that the following sections are
illustrative enough, to the point that the reader is able to easily categorise any
future Geometric Deep Learning developments using the lens of invariances
and symmetries.


--- Page 73 ---
5. GEOMETRIC DEEP LEARNING MODELS
69
0 1 1 1 0 0 0
0 0 1 1 1 0 0
0 0 0 1 1 1 0
0 0 0 1 1 0 0
0 0 1 1 0 0 0
0 1 1 0 0 0 0
1 1 0 0 0 0 0
x
⋆
1 0 1
0 1 0
1 0 1
C(θ)
z
}|
{
θ11 + θ13 + θ22 + θ31 + θ33
=
1 4 3 4 1
1 2 4 3 3
1 2 3 4 1
1 3 3 1 1
3 3 1 1 0
x ⋆θ
1 0 1
0 1 0
1 0 1
×1
×0
×1
×0
×1
×0
×1
×0
×1
Figure 14: The process of convolving an image x with a ﬁlter C(θ). The ﬁlter
parameters θ can be expressed as a linear combination of generators θvw.
5.1
Convolutional Neural Networks
Convolutional Neural Networks are perhaps the earliest and most well
known example of deep learning architectures following the blueprint of
Geometric Deep Learning outlined in Section 3.5. In Section 4.2 we have
fully characterised the class of linear and local translation equivariant op-
erators, given by convolutions C(θ)x = x ⋆θ with a localised ﬁlter θ
Recall, C(θ) is a circulant
matrix with parameters θ.
. Let
us ﬁrst focus on scalar-valued (‘single-channel’ or ‘grayscale’) discretised
images, where the domain is the grid Ω= [H] × [W] with u = (u1, u2) and
x ∈X(Ω, R).
Any convolution with a compactly supported ﬁlter of size Hf × W f can
be written as a linear combination of generators θ1,1, . . . , θHf,W f , given for
example by the unit peaks θvw(u1, u2) = δ(u1 −v, u2 −w). Any local linear
equivariant map is thus expressible as
Note that we usually imagine
x and θvw as 2D matrices, but
in this equation, both x and
θvw have their two coordinate
dimensions ﬂattened into
one—making x a vector, and
C(θvw) a matrix.
F(x) =
Hf
X
v=1
W f
X
w=1
αvwC(θvw)x ,
(26)
which, in coordinates, corresponds to the familiar 2D convolution (see Figure
14 for an overview):
F(x)uv =
Hf
X
a=1
W f
X
b=1
αabxu+a,v+b .
(27)


--- Page 74 ---
70
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
Other choices of the basis θvw are also possible and will yield equivalent
operations (for potentially diﬀerent choices of αvw). A popular example are
directional derivatives: θvw(u1, u2) = δ(u1, u2)−δ(u1−v, u2−w), (v, w) ̸= (0, 0)
taken together with the local average θ0(u1, u2) =
1
HfWf . In fact, directional
derivatives can be considered a grid-speciﬁc analogue of diﬀusion processes
on graphs, which we recover if we assume each pixel to be a node connected
to its immediate neighbouring pixels in the grid.
When the scalar input channel is replaced by multiple channels (e.g., RGB
colours, or more generally an arbitrary number of feature maps), the con-
volutional ﬁlter becomes a convolutional tensor expressing arbitrary linear
combinations of input features into output feature maps. In coordinates, this
can be expressed as
F(x)uvj =
Hf
X
a=1
W f
X
b=1
M
X
c=1
αjabcxu+a,v+b,c , j ∈[N] ,
(28)
where M and N are respectively the number of input and output channels.
This basic operation encompasses a broad class of neural network archi-
tectures, which, as we will show in the next section, have had a profound
impact across many areas of computer vision, signal processing, and beyond.
Here, rather than dissecting the myriad of possible architectural variants of
CNNs, we prefer to focus on some of the essential innovations that enabled
their widespread use.
Eﬃcient multiscale computation
As discussed in the GDL template for
general symmetries, extracting translation invariant features out of the con-
volutional operator F requires a non-linear step.
−3
−2
−1
0
1
2
3
0
1
2
3
ReLU, often considered a
‘modern’ architectural choice,
was already used in the
Neocognitron (Fukushima
and Miyake, 1982).
Rectiﬁcation is equivalent to
the principle of
demodulation, which is
fundamental in electrical
engineering as the basis for
many transmission protocols,
such as FM radio; and also
has a prominent role in
models for neuronal activity.
Convolutional features are
processed through a non-linear activation function σ, acting element-wise on
the input—i.e., σ : X(Ω) →X(Ω), as σ(x)(u) = σ(x(u)). Perhaps the most
popular example at the time of writing is the Rectiﬁed Linear Unit (ReLU):
σ(x) = max(x, 0). This non-linearity eﬀectively rectiﬁes the signals, pushing
their energy towards lower frequencies, and enabling the computation of
high-order interactions across scales by iterating the construction.
Already in the early works of Fukushima and Miyake (1982) and LeCun
et al. (1998), CNNs and similar architectures had a multiscale structure,
where after each convolutional layer (28) one performs a grid coarsening P :
X(Ω) →X(Ω′), where the grid Ω′ has coarser resolution than Ω. This enables


--- Page 75 ---
5. GEOMETRIC DEEP LEARNING MODELS
71
multiscale ﬁlters with eﬀectively increasing receptive ﬁeld, yet retaining a
constant number of parameters per scale. Several signal coarsening strategies
P (referred to as pooling) may be used, the most common are applying a low-
pass anti-aliasing ﬁlter (e.g. local average) followed by grid downsampling,
or non-linear max-pooling.
In summary, a ‘vanilla’ CNN layer can be expressed as the composition of the
basic objects already introduced in our Geometric Deep Learning blueprint:
h = P(σ(F(x))) ,
(29)
i.e. an equivariant linear layer F, a coarsening operation P, and a non-
linearity σ. It is also possible to perform translation invariant global pooling
operations within CNNs. Intuitively, this involves each pixel—which, after
several convolutions, summarises a patch centered around it—proposing the
ﬁnal representation of the image
CNNs which only consist of
the operations mentioned in
this paragraph are often
dubbed “all-convolutional”.
In contrast, many CNNs
ﬂatten the image across the
spatial axes and pass them to
an MLP classiﬁer, once
suﬃcient equivariant and
coarsening layers have been
applied. This loses
translation invariance.
, with the ultimate choice being guided by a
form of aggregation of these proposals. A popular choice here is the average
function, as its outputs will retain similar magnitudes irrespective of the
image size (Springenberg et al., 2014).
Prominent examples following this CNN blueprint (some of which we will
discuss next) are displayed in Figure 15.
Deep and Residual Networks
A CNN architecture, in its simplest form, is
therefore speciﬁed by hyperparameters (Hf
k , W f
k , Nk, pk)k≤K, with Mk+1 =
Nk and pk = 0, 1 indicating whether grid coarsening is performed or not.
While all these hyperparameters are important in practice, a particularly
important question is to understand the role of depth K in CNN architectures,
and what are the fundamental tradeoﬀs involved in choosing such a key
hyperparameter, especially in relation to the ﬁlter sizes (Hf
k , W f
k ).
While a rigorous answer to this question is still elusive, mounting empirical
evidence collected throughout the recent years suggests a favourable tradeoﬀ
towards deeper (large K) yet thinner (small (Hf
k , W f
k )) models
Historically, ResNet models
are predated by highway
networks (Srivastava et al.,
2015), which allow for more
general gating mechanisms to
control the residual
information ﬂow.
. In this
context, a crucial insight by He et al. (2016) was to reparametrise each
convolutional layer to model a perturbation of the previous features, rather
than a generic non-linear transformation:
h = P (x + σ(F(x))) .
(30)
The resulting residual networks provide several key advantages over the
previous formulation. In essence, the residual parametrisation is consistent


--- Page 76 ---
72
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
1
32
6
28
16
10
1
120
1
84
10
SOFT
3
224
96
55
256
27
384
13
384
13
256
13
1
4096
1
4096
1000
SOFT
1
28
28
Input
16
28
3x3
Conv
16
28
16
28
+
16
28
16
28
16
28
+
16
28
16
28
16
28
+
16
28
32
28
32
28
1x1
32
28
BatchNorm
ReLU
+
32
28
128
23
23
Average pooling
1
10
SOFT
6464
I
128 128
I/2
256
256
I/4
512
512
I/8
1024
1024
I/16
Bottleneck Conv
512
512
512
512
I/8
256
256
256
256
I/4
128 128 128 128
I/2
64 64 64 64
I
Softmax
Figure 15: Prominent examples of CNN architectures.
Top-to-bottom:
LeNet (LeCun et al., 1998), AlexNet (Krizhevsky et al., 2012), ResNet (He
et al., 2016) and U-Net (Ronneberger et al., 2015). Drawn using the PlotNeu-
ralNet package (Iqbal, 2018).


--- Page 77 ---
5. GEOMETRIC DEEP LEARNING MODELS
73
with the view that the deep network is a discretisation of an underlying
continuous dynamical system, modelled as an ordinary diﬀerential equation
(ODE)
In this case, the ResNet is
performing a Forward Euler
discretisation of an ODE:
˙x = σ(F(x))
. Crucially, learning a dynamical system by modeling its velocity turns
out to be much easier than learning its position directly. In our learning
setup, this translates into an optimisation landscape with more favorable
geometry, leading to the ability to train much deeper architectures than was
possible before. As will be discussed in future work, learning using deep
neural networks deﬁnes a non-convex optimisation problem, which can be
eﬃciently solved using gradient-descent methods under certain simplifying
regimes. The key advantage of the ResNet parametrisation has been rigor-
ously analysed in simple scenarios (Hardt and Ma, 2016), and remains an
active area of theoretical investigation. Finally, Neural ODEs (Chen et al.,
2018) are a recent popular architecture that pushes the analogy with ODEs
even further, by learning the parameters of the ODE ˙x = σ(F(x)) directly
and relying on standard numerical integration.
Normalisation
Another important algorithmic innovation that boosted the
empirical performance of CNNs signiﬁcantly is the notion of normalisation.
In early models of neural activity, it was hypothesised that neurons perform
some form of local ‘gain control’, where the layer coeﬃcients xk are replaced
by ˜xk = σ−1
k
⊙(xk −µk). Here, µk and σk encode the ﬁrst and second-
order moment information of xk, respectively. Further, they can be either
computed globally or locally.
In the context of Deep Learning, this principle was widely adopted through
the batch normalisation layer (Ioﬀe and Szegedy, 2015)
We note that normalising
activations of neural
networks has seen attention
even before the advent of
batch normalisation. See, e.g.,
Lyu and Simoncelli (2008).
, followed by several
variants (Ba et al., 2016; Salimans and Kingma, 2016; Ulyanov et al., 2016;
Cooijmans et al., 2016; Wu and He, 2018). Despite some attempts to rigor-
ously explain the beneﬁts of normalisation in terms of better conditioned
optimisation landscapes (Santurkar et al., 2018), a general theory that can
provide guiding principles is still missing at the time of writing.
Data augmentation
While CNNs encode the geometric priors associated
with translation invariance and scale separation, they do not explicitly ac-
count for other known transformations that preserve semantic information,
e.g lightning or color changes, or small rotations and dilations. A pragmatic
approach to incorporate these priors with minimal architectural changes is
to perform data augmentation, where one manually performs said transfor-


--- Page 78 ---
74
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
mations to the input images and adds them into the training set.
Data augmentation has been empirically successful and is widely used—not
only to train state-of-the-art vision architectures, but also to prop up several
developments in self-supervised and causal representation learning (Chen
et al., 2020; Grill et al., 2020; Mitrovic et al., 2020). However, it is provably
sub-optimal in terms of sample complexity (Mei et al., 2021); a more eﬃcient
strategy considers instead architectures with richer invariance groups—as
we discuss next.
5.2
Group-equivariant CNNs
As discussed in Section 4.3, we can generalise the convolution operation
from signals on a Euclidean space to signals on any homogeneous space Ω
acted upon by a group G
Recall that a homogeneous
space is a set Ωequipped
with a transitive group action,
meaning that for any u, v ∈Ω
there exists g ∈G such that
g.u = v.
. By analogy to the Euclidean convolution where
a translated ﬁlter is matched with the signal, the idea of group convolution
is to move the ﬁlter around the domain using the group action, e.g. by
rotating and translating. By virtue of the transitivity of the group action,
we can move the ﬁlter to any position on Ω. In this section, we will discuss
several concrete examples of the general idea of group convolution, including
implementation aspects and architectural choices.
Discrete group convolution
We begin by considering the case where the
domain Ωas well as the group G are discrete. As our ﬁrst example, we
consider medical volumetric images represented as signals of on 3D grids
with discrete translation and rotation symmetries. The domain is the 3D
cubical grid Ω= Z3 and the images (e.g. MRI or CT 3D scans) are modelled
as functions x : Z3 →R, i.e. x ∈X(Ω). Although in practice such images
have support on a ﬁnite cuboid [W] × [H] × [D] ⊂Z3, we instead prefer
to view them as functions on Z3 with appropriate zero padding. As our
symmetry, we consider the group G = Z3 ⋊Oh of distance- and orientation-
preserving transformations on Z3. This group consists of translations (Z3)
and the discrete rotations Oh generated by 90 degree rotations about the
three axes (see Figure 16).
As our second example, we consider DNA
DNA is a biopolymer
molecule made of four
repeating units called
nucleotides (Cytosine,
Guanine, Adenine, and
Thymine), arranged into two
strands coiled around each
other in a double helix,
where each nucleotide occurs
opposite of the
complementary one (base
pairs A/T and C/G).
sequences made up of four letters:
C, G, A, and T. The sequences can be represented on the 1D grid Ω= Z as
signals x : Z →R4, where each letter is one-hot coded in R4. Naturally, we


--- Page 79 ---
5. GEOMETRIC DEEP LEARNING MODELS
75
Figure 16: A 3 × 3 ﬁlter, rotated by all 24 elements of the discrete rotation
group Oh, generated by 90-degree rotations about the vertical axis (red
arrows), and 120-degree rotations about a diagonal axis (blue arrows).
have a discrete 1D translation symmetry on the grid, but DNA sequences
have an additional interesting symmetry. This symmetry arises from the
way DNA is physically embodied as a double helix, and the way it is read by
the molecular machinery of the cell. Each strand of the double helix begins
with what is called the 5′-end and ends with a 3′-end, with the 5′ on one
strand complemented by a 3′ on the other strand. In other words, the two
strands have an opposite orientation.
3’
5’
5’
3’
C
G
A
T
T
A
T
A
C
G
C
G
T
A
T
A
G
C
G
C
G
C
T
A
A schematic of the DNA’s
double helix structure, with
the two strands coloured in
blue and red. Note how the
sequences in the helices are
complementary and read in
reverse (from 5’ to 3’).
Since the DNA molecule is always read
oﬀstarting at the 5′-end, but we do not know which one, a sequence such as
ACCCTGG is equivalent to the reversed sequence with each letter replaced
by its complement, CCAGGGT. This is called reverse-complement symmetry
of the letter sequence. We thus have the two-element group Z2 = {0, 1}
corresponding to the identity 0 and reverse-complement transformation 1
(and composition 1 + 1 = 0 mod 2). The full group combines translations
and reverse-complement transformations.
In our case, the group convolution (14) we deﬁned in Section 4.3 is given as
(x ⋆θ)(g) =
X
u∈Ω
xuρ(g)θu,
(31)
the inner product between the (single-channel) input signal x and a ﬁlter θ
transformed by g ∈G via ρ(g)θu = θg−1u, and the output x ⋆θ is a function


--- Page 80 ---
76
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
on G. Note that since Ωis discrete, we have replaced the integral from
equation (14) by a sum.
Transform+Convolve approach
We will show that the group convolution
can be implemented in two steps: a ﬁlter transformation step, and a transla-
tional convolution step. The ﬁlter transformation step consists of creating
rotated (or reverse-complement transformed) copies of a basic ﬁlter, while
the translational convolution is the same as in standard CNNs and thus
eﬃciently computable on hardware such as GPUs. To see this, note that
in both of our examples we can write a general transformation g ∈G as a
transformation h ∈H (e.g. a rotation or reverse-complement transformation)
followed by a translation k ∈Zd, i.e. g = kh (with juxtaposition denoting
the composition of the group elements k and h). By properties of the group
representation, we have ρ(g) = ρ(kh) = ρ(k)ρ(h). Thus,
(x ⋆θ)(kh) =
X
u∈Ω
xuρ(k)ρ(h)θu
=
X
u∈Ω
xu(ρ(h)θ)u−k
(32)
We recognise the last equation as the standard (planar Euclidean) convo-
lution of the signal x and the transformed ﬁlter ρ(h)θ. Thus, to implement
group convolution for these groups, we take the canonical ﬁlter θ, create
transformed copies θh = ρ(h)θ for each h ∈H (e.g. each rotation h ∈Oh
or reverse-complement DNA symmetry h ∈Z2), and then convolve x with
each of these ﬁlters: (x ⋆θ)(kh) = (x ⋆θh)(k). For both of our examples,
the symmetries act on ﬁlters by simply permuting the ﬁlter coeﬃcients, as
shown in Figure 16 for discrete rotations. Hence, these operations can be
implemented eﬃciently using an indexing operation with pre-computed
indices.
While we deﬁned the feature maps output by the group convolution x ⋆θ
as functions on G, the fact that we can split g into h and k means that we
can also think of them as a stack of Euclidean feature maps (sometimes
called orientation channels), with one feature map per ﬁlter transformation /
orientation k. For instance, in our ﬁrst example we would associate to each
ﬁlter rotation (each node in Figure 16) a feature map, which is obtained by
convolving (in the traditional translational sense) the rotated ﬁlter. These
feature maps can thus still be stored as a W ×H ×C array, where the number


--- Page 81 ---
5. GEOMETRIC DEEP LEARNING MODELS
77
of channels C equals the number of independent ﬁlters times the number of
transformations h ∈H (e.g. rotations).
As shown in Section 4.3, the group convolution is equivariant: (ρ(g)x) ⋆θ =
ρ(g)(x ⋆θ). What this means in terms of orientation channels is that under
the action of h, each orientation channel is transformed, and the orientation
channels themselves are permuted. For instance, if we associate one orien-
tation channel per transformation in Figure 16 and apply a rotation by 90
degrees about the z-axis (corresponding to the red arrows), the feature maps
will be permuted as shown by the red arrows. This description makes it clear
that a group convolutional neural network bears much similarity to a tradi-
tional CNN. Hence, many of the network design patterns discussed in the
Section 5.1, such as residual networks, can be used with group convolutions
as well.
Spherical CNNs in the Fourier domain
For the continuous symmetry
group of the sphere that we saw in Section 4.3, it is possible to implement the
convolution in the spectral domain, using the appropriate Fourier transform
(we remind the reader that the convolution on S2 is a function on SO(3),
hence we need to deﬁne the Fourier transform on both these domains in
order to implement multi-layer spherical CNNs). Spherical harmonics are an
orthogonal basis on the 2D sphere, analogous to the classical Fourier basis
of complex exponential. On the special orthogonal group, the Fourier basis
is known as the Wigner D-functions. In both cases, the Fourier transforms
(coeﬃcients) are computed as the inner product with the basis functions,
and an analogy of the Convolution Theorem holds: one can compute the
convolution in the Fourier domain as the element-wise product of the Fourier
transforms. Furthermore, FFT-like algorithms exist for the eﬃcient compu-
tation of Fourier transform on S2 and SO(3). We refer for further details to
Cohen et al. (2018).
5.3
Graph Neural Networks
Graph Neural Networks (GNNs) are the realisation of our Geometric Deep
Learning blueprint on graphs leveraging the properties of the permutation
group. GNNs are among the most general class of deep learning architec-
tures currently in existence, and as we will see in this text, most other deep
learning architectures can be understood as a special case of the GNN with


--- Page 82 ---
78
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
cba
cbc
cbd
cbe
cbb
Convolutional
xb
xa
xc
xd
xe
αba
αbc
αbd
αbe
αbb
Attentional
xb
xa
xc
xd
xe
mba
mbc
mbd
mbe
mbb
Message-passing
xb
xa
xc
xd
xe
Figure 17: A visualisation of the dataﬂow for the three ﬂavours of GNN
layers, g. We use the neighbourhood of node b from Figure 10 to illustrate
this. Left-to-right: convolutional, where sender node features are multiplied
with a constant, cuv; attentional, where this multiplier is implicitly computed
via an attention mechanism of the receiver over the sender: αuv = a(xu, xv);
and message-passing, where vector-based messages are computed based
on both the sender and receiver: muv = ψ(xu, xv).
additional geometric structure.
As per our discussion in Section 4.1, we consider a graph to be speciﬁed
with an adjacency matrix A and node features X. We will study GNN
architectures that are permutation equivariant functions F(X, A) constructed
by applying shared permutation invariant functions φ(xu, XNu) over local
neighbourhoods. Under various guises, this local function φ can be referred
to as “diﬀusion”, “propagation”, or “message passing”, and the overall
computation of such F as a “GNN layer”.
The design and study of GNN layers is one of the most active areas of deep
learning at the time of writing, making it a landscape that is challenging to
navigate. Fortunately, we ﬁnd that the vast majority of the literature may be
derived from only three “ﬂavours” of GNN layers (Figure 17), which we will
present here. These ﬂavours govern the extent to which φ transforms the
neighbourhood features, allowing for varying degrees of complexity when
modelling interactions across the graph.
In all three ﬂavours, permutation invariance is ensured by aggregating fea-
tures from XNu (potentially transformed, by means of some function ψ)
with some permutation-invariant function L, and then updating the features
of node u, by means of some function φ. Typically,
Most commonly, ψ and φ are
learnable aﬃne
transformations with
activation functions; e.g.
ψ(x) = Wx + b;
φ(x, z) = σ (Wx + Uz + b),
where W, U, b are learnable
parameters and σ is an
activation function such as
the rectiﬁed linear unit. The
additional input of xu to φ
represents an optional
skip-connection, which is often
very useful.
ψ and φ are learnable,
whereas L is realised as a nonparametric operation such as sum, mean, or
maximum, though it can also be constructed e.g. using recurrent neural
networks (Murphy et al., 2018).


--- Page 83 ---
5. GEOMETRIC DEEP LEARNING MODELS
79
In the convolutional ﬂavour (Kipf and Welling, 2016a; Deﬀerrard et al.,
2016; Wu et al., 2019), the features of the neighbourhood nodes are directly
aggregated with ﬁxed weights,
hu = φ
 
xu,
M
v∈Nu
cuvψ(xv)
!
.
(33)
Here, cuv speciﬁes the importance of node v to node u’s representation. It
is a constant that often directly depends on the entries in A representing
the structure of the graph. Note that when the aggregation operator L
is chosen to be the summation, it can be considered as a linear diﬀusion
or position-dependent linear ﬁltering, a generalisation of convolution.
It is worthy to note that this
ﬂavour does not express every
GNN layer that is
convolutional (in the sense of
commuting with the graph
structure), but covers most
such approaches proposed in
practice. We will provide
detailed discussion and
extensions in future work.
In
particular, the spectral ﬁlters we have seen in Sections 4.4 and 4.6 fall under
this category, as they amount to applying ﬁxed local operators (e.g. the
Laplacian matrix) to node-wise signals.
In the attentional ﬂavour (Veličković et al., 2018; Monti et al., 2017; Zhang
et al., 2018), the interactions are implicit
hu = φ
 
xu,
M
v∈Nu
a(xu, xv)ψ(xv)
!
.
(34)
Here, a is a learnable self-attention mechanism that computes the importance
coeﬃcients αuv = a(xu, xv) implicitly. They are often softmax-normalised
across all neighbours. When L is the summation, the aggregation is still a
linear combination of the neighbourhood node features, but now the weights
are feature-dependent.
Finally, the message-passing ﬂavour (Gilmer et al., 2017; Battaglia et al.,
2018) amounts to computing arbitrary vectors (“messages”) across edges,
hu = φ
 
xu,
M
v∈Nu
ψ(xu, xv)
!
.
(35)
Here, ψ is a learnable message function, computing v’s vector sent to u, and the
aggregation can be considered as a form of message passing on the graph.
One important thing to note is a representational containment between these
approaches: convolution ⊆attention ⊆message-passing. Indeed, attentional
GNNs can represent convolutional GNNs by an attention mechanism im-
plemented as a look-up table a(xu, xv) = cuv, and both convolutional and


--- Page 84 ---
80
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
attentional GNNs are special cases of message-passing where the messages
are only the sender nodes’ features: ψ(xu, xv) = cuvψ(xv) for convolutional
GNNs and ψ(xu, xv) = a(xu, xv)ψ(xv) for attentional GNNs.
This does not imply that message passing GNNs are always the most use-
ful variant; as they have to compute vector-valued messages across edges,
they are typically harder to train and require unwieldy amounts of memory.
Further, on a wide range of naturally-occurring graphs, the graph’s edges
encode for downstream class similarity (i.e. an edge (u, v) implies that u
and v are likely to have the same output). For such graphs (often called
homophilous), convolutional aggregation across neighbourhoods is often a
far better choice, both in terms of regularisation and scalability. Attentional
GNNs oﬀer a “middle-ground”: they allow for modelling complex interac-
tions within neighbourhoods while computing only scalar-valued quantities
across the edges, making them more scalable than message-passing.
The “three ﬂavour” categorisation presented here is provided with brevity in
mind and inevitably neglects a wealth of nuances, insights, generalisations,
and historical contexts to GNN models. Importantly, it excludes higher-
dimensional GNN based on the Weisfeiler-Lehman hierarchy and spectral
GNNs relying on the explicit computation of the graph Fourier transform.
5.4
Deep Sets, Transformers, and Latent Graph Inference
We close the discussion on GNNs by remarking on permutation-equivariant
neural network architectures for learning representations of unordered sets.
While sets have the least structure among the domains we have discussed in
this text, their importance has been recently highlighted by highly-popular
architectures such as Transformers (Vaswani et al., 2017) and Deep Sets
(Zaheer et al., 2017). In the language of Section 4.1, we assume that we are
given a matrix of node features, X, but without any speciﬁed adjacency or
ordering information between the nodes. The speciﬁc architectures will arise
by deciding to what extent to model interactions between the nodes.
Empty edge set
Unordered sets are provided without any additional struc-
ture or geometry whatsoever—hence, it could be argued that the most natural
way to process them is to treat each set element entirely independently. This
translates to a permutation equivariant function over such inputs, which


--- Page 85 ---
5. GEOMETRIC DEEP LEARNING MODELS
81
was already introduced in Section 4.1: a shared transformation applied to
every node in isolation. Assuming the same notation as when describing
GNNs (Section 5.3), such models can be represented as
hu = ψ(xu),
where ψ is a learnable transformation. It may be observed that this is a
special case of a convolutional GNN with Nu = {u}—or, equivalently, A = I.
Such an architecture is commonly referred to as Deep Sets, in recognition
of the work of Zaheer et al. (2017) that have theoretically proved several
universal-approximation properties of such architectures. It should be noted
that the need to process unordered sets commonly arises in computer vision
and graphics when dealing with point clouds; therein, such models are known
as PointNets (Qi et al., 2017).
Complete edge set
While assuming an empty edge set is a very eﬃcient
construct for building functions over unordered sets, often we would expect
that elements of the set exhibit some form of relational structure—i.e., that
there exists a latent graph between the nodes. Setting A = I discards any such
structure, and may yield suboptimal performance. Conversely, we could
assume that, in absence of any other prior knowledge, we cannot upfront
exclude any possible links between nodes. In this approach we assume the
complete graph, A = 11⊤; equivalently, Nu = V. As we do not assume access
to any coeﬃcients of interaction, running convolutional-type GNNs over such
a graph would amount to:
hu = φ
 
xu,
M
v∈V
ψ(xv)
!
,
where the second input, L
v∈V ψ(xv) is identical for all nodes u
This is a direct consequence
of the permutation invariance
of L.
, and as such
makes the model equivalently expressive to ignoring that input altogether;
i.e. the A = I case mentioned above.
This motivates the use of a more expressive GNN ﬂavour, the attentional,
hu = φ
 
xu,
M
v∈V
a(xu, xv)ψ(xv)
!
(36)
which yields the self-attention operator, the core of the Transformer archi-
tecture (Vaswani et al., 2017). Assuming some kind of normalisation over


--- Page 86 ---
82
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
the attentional coeﬃcients (e.g. softmax), we can constrain all the scalars
a(xu, xv) to be in the range [0, 1]; as such, we can think of self-attention as
inferring a soft adjacency matrix, auv = a(xu, xv), as a byproduct of gradient-
based optimisation for some downstream task.
The above perspective means that we can pose Transformers exactly as at-
tentional GNNs over a complete graph (Joshi, 2020).
It is also appropriate to apply
the message-passing ﬂavour.
While popular for physics
simulations and relational
reasoning (e.g. Battaglia et al.
(2016); Santoro et al. (2017)),
they have not been as widely
used as Transformers. This is
likely due to the memory
issues associated with
computing vector messages
over a complete graph, or the
fact that vector-based
messages are less
interpretable than the “soft
adjacency” provided by
self-attention.
However, this is in
apparent conﬂict with Transformers being initially proposed for modelling
sequences—the representations of hu should be mindful of node u’s position in
the sequence, which complete-graph aggregation would ignore. Transform-
ers address this issue by introducing positional encodings: the node features
xu are augmented to encode node u’s position in the sequence, typically as
samples from a sine wave whose frequency is dependent on u.
On graphs, where no natural ordering of nodes exists, multiple alternatives
were proposed to such positional encodings. While we defer discussing
these alternatives for later, we note that one promising direction involves a
realisation that the positional encodings used in Transformers can be directly
related to the discrete Fourier transform (DFT), and hence to the eigenvectors
of the graph Laplacian of a “circular grid”. Hence, Transformers’ positional
encodings are implicitly representing our assumption that input nodes are
connected in a grid. For more general graph structures, one may simply
use the Laplacian eigenvectors of the (assumed) graph—an observation
exploited by Dwivedi and Bresson (2020) within their empirically powerful
Graph Transformer model.
Inferred edge set
Finally, one can try to learn the latent relational structure,
leading to some general A that is neither I nor 11⊤. The problem of inferring
a latent adjacency matrix A for a GNN to use (often called latent graph
inference) is of high interest for graph representation learning. This is due
to the fact that assuming A = I may be representationally inferior, and
A = 11⊤may be challenging to implement due to memory requirements
and large neighbourhoods to aggregate over. Additionally, it is closest to the
“true” problem: inferring an adjacency matrix A implies detecting useful
structure between the rows of X, which may then help formulate hypotheses
such as causal relations between variables.
Unfortunately, such a framing necessarily induces a step-up in modelling
complexity. Speciﬁcally, it requires properly balancing a structure learning


--- Page 87 ---
5. GEOMETRIC DEEP LEARNING MODELS
83
objective (which is discrete, and hence challenging for gradient-based optimi-
sation) with any downstream task the graph is used for. This makes latent
graph inference a highly challenging and intricate problem.
5.5
Equivariant Message Passing Networks
In many applications of Graph Neural Networks, node features (or parts
thereof) are not just arbitrary vectors but coordinates of geometric entities.
This is the case, for example, when dealing with molecular graphs: the nodes
representing atoms may contain information about the atom type as well
as its 3D spatial coordinates. It is desirable to process the latter part of the
features in a manner that would transform in the same way as the molecule is
transformed in space, in other words, be equivariant to the Euclidean group
E(3) of rigid motions (rotations, translations, and reﬂections) in addition to
the standard permutation equivariance discussed before.
To set the stage for our (slightly simpliﬁed) analysis, we will make a distinc-
tion between node features fu ∈Rd and node spatial coordinates xu ∈R3; the
latter are endowed with Euclidean symmetry structure. In this setting, an
equivariant layer explicitly transforms these two inputs separately, yielding
modiﬁed node features f′
u and coordinates x′
u.
We can now state our desirable equivariance property, following the Geo-
metric Deep Learning blueprint. If the spatial component of the input is
transformed by g ∈E(3) (represented as ρ(g)x = Rx + b, where R is an
orthogonal matrix modeling rotations and reﬂections, and b is a translation
vector), the spatial component of the output transforms in the same way (as
x′
u 7→Rx′
u + b), whereas f′
u remains invariant.
Much like the space of permutation equivariant functions we discussed
before in the context of general graphs, there exists a vast amount of E(3)-
equivariant layers that would satisfy the constraints above—but not all of
these layers would be geometrically stable, or easy to implement. In fact, the
space of practically useful equivariant layers may well be easily described
by a simple categorisation, not unlike our “three ﬂavours” of spatial GNN
layers. One elegant solution was suggested by Satorras et al. (2021) in the


--- Page 88 ---
84
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
form of equivariant message passing. Their model operates as follows:
f′
u
=
φ
 
fu,
M
v∈Nu
ψf(fu, fv, ∥xu −xv∥2)
!
,
x′
u
=
xu +
X
v̸=u
(xu −xv)ψc(fu, fv, ∥xu −xv∥2)
where ψf and ψc are two distinct (learnable) functions. It can be shown that
such an aggregation is equivariant under Euclidean transformations of the
spatial coordinates. This is due to the fact that the only dependence of f′
u on
xu is through the distances ∥xu −xv∥2, and the action of E(3) necessarily
leaves distances between nodes unchanged. Further, the computations of
such a layer can be seen as a particular instance of the “message-passing”
GNN ﬂavour, hence they are eﬃcient to implement.
To summarise, in contrast to ordinary GNNs, Satorras et al. (2021) enable
the correct treatment of ‘coordinates’ for each point in the graph. They
are now treated as a member of the E(3) group, which means the network
outputs behave correctly under rotations, reﬂections and translations of the
input.
90°
While scalar features
(heatmap) do not change
under rotations, vector
features (arrows) may
change direction. The simple
E(3) equivariant GNN given
before does not take this into
account.
The features, fu, however, are treated in a channel-wise manner and
still assumed to be scalars that do not change under these transformations.
This limits the type of spatial information that can be captured within such
a framework. For example, it may be desirable for some features to be
encoded as vectors—e.g. point velocities—which should change direction
under such transformations. Satorras et al. (2021) partially alleviate this issue
by introducing the concept of velocities in one variant of their architecture.
Velocities are a 3D vector property of each point which rotates appropriately.
However, this is only a small subspace of the general representations that
could be learned with an E(3) equivariant network. In general, node features
may encode tensors of arbitrary dimensionality that would still transform
according to E(3) in a well-deﬁned manner.
Hence, while the architecture discussed above already presents an elegant
equivariant solution for many practical input representations, in some cases
it may be desirable to explore a broader collection of functions that satisfy
the equivariance property. Existing methods dealing with such settings can
be categorised into two classes: irreducible representations (of which the pre-
viously mentioned layer is a simpliﬁed instance) and regular representations.
We brieﬂy survey them here, leaving detailed discussion to future work.


--- Page 89 ---
5. GEOMETRIC DEEP LEARNING MODELS
85
Irreducible representations
Irreducible representations build on the ﬁnd-
ing that all elements of the roto-translation group can be brought into an
irreducible form: a vector that is rotated by a block diagonal matrix. Cru-
cially, each of those blocks is a Wigner D-matrix (the aforementioned Fourier
basis for Spherical CNNs). Approaches under this umbrella map from one
set of irreducible representations to another using equivariant kernels. To
ﬁnd the full set of equivariant mappings, one can then directly solve the
equivariance constraint over these kernels. The solutions form a linear com-
bination of equivariant basis matrices derived by Clebsch-Gordan matrices and
the spherical harmonics.
Early examples of the irreducible representations approach include Tensor
Field Networks (Thomas et al., 2018) and 3D Steerable CNNs (Weiler et al.,
2018), both convolutional models operating on point clouds. The SE(3)-
Transformer of Fuchs et al. (2020) extends this framework to the graph
domain, using an attentional layer rather than convolutional. Further, while
our discussion focused on the special case solution of Satorras et al. (2021),
we note that the motivation for rotation or translation equivariant predic-
tions over graphs had historically been explored in other ﬁelds, including
architectures such as Dynamic Graph CNN (Wang et al., 2019b) for point
clouds and eﬃcient message passing models for quantum chemistry, such
as SchNet (Schütt et al., 2018) and DimeNet (Klicpera et al., 2020).
Regular representations
While the approach of irreducible representa-
tions is attractive, it requires directly reasoning about the underlying group
representations, which may be tedious, and only applicable to groups that
are compact. Regular representation approaches are more general, but come
with an additional computational burden – for exact equivariance they re-
quire storing copies of latent feature embeddings for all group elements
This approach was, in fact,
pioneered by the group
convolutional neural
networks we presented in
previous sections.
.
One promising approach in this space aims to observe equivariance to Lie
groups—through deﬁnitions of exponential and logarithmic maps—with the
promise of rapid prototyping across various symmetry groups. While Lie
groups are out of scope for this section, we refer the reader to two recent
successful instances of this direction: the LieConv of Finzi et al. (2020), and
the LieTransformer of Hutchinson et al. (2020).
The approaches covered in this section represent popular ways of processing
data on geometric graphs in an way that is explicitly equivariant to the un-


--- Page 90 ---
86
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
derlying geometry. As discussed in Section 4.6, meshes are a special instance
of geometric graphs that can be understood as discretisations of continuous
surfaces. We will study mesh-speciﬁc equivariant neural networks next.
5.6
Intrinsic Mesh CNNs
Meshes, in particular, triangular ones, are the ‘bread and butter’ of computer
graphics and perhaps the most common way of modeling 3D objects. The
remarkable success of deep learning in general and CNNs in computer vision
in particular has lead to a keen interest in the graphics and geometry pro-
cessing community around the mid-2010s
Examples of geodesic patches.
In order for the resulting
patch to be a topological disk,
its radius R must be smaller
than the injectivity radius.
to construct similar architectures
for mesh data.
Geodesic patches
Most of the architectures for deep learning on meshes
implement convolutional ﬁlters of the form (21) by discretising or approxi-
mating the exponential map and expressing the ﬁlter in a coordinate system
of the tangent plane. Shooting a geodesic γ : [0, T] →Ωfrom a point u = γ(0)
to nearby point v = γ(T) deﬁnes a local system of geodesic polar coordinates
(r(u, v), ϑ(u, v)) where r is the geodesic distance between u and v (length of
the geodesic γ) and ϑ is the angle between γ′(0) and some local reference
direction. This allows to deﬁne a geodesic patch x(u, r, ϑ) = x(expu ˜ω(r, ϑ)),
where ˜ωu : [0, R] × [0, 2π) →TuΩis the local polar frame.
On a surface
Construction of discrete
geodesics on a mesh.
discretised as a mesh, a geodesic is a poly-line that traverses
the triangular faces. Traditionally, geodesics have been computed using the
Fast Marching algorithm Kimmel and Sethian (1998), an eﬃcient numerical
approximation of a nonlinear PDE called the eikonal equation encountered in
physical models of wave propagation in a medium. This scheme was adapted
by Kokkinos et al. (2012) for the computation of local geodesic patches and
later reused by Masci et al. (2015) for the construction of Geodesic CNNs, the
ﬁrst intrinsic CNN-like architectures on meshes.
Isotropic ﬁlters
Importantly, in the deﬁnition of the geodesic patch we have
ambiguity in the choice of the reference direction and the patch orientation.
This is exactly the ambiguity of the choice of the gauge, and our local system
of coordinates is deﬁned up to arbitrary rotation (or a shift in the angular
coordinate, x(u, r, ϑ + ϑ0)), which can be diﬀerent at every node. Perhaps


--- Page 91 ---
5. GEOMETRIC DEEP LEARNING MODELS
87
the most straightforward solution is to use isotropic ﬁlters of the form θ(r)
that perform a direction-independent aggregation of the neighbour features,
(x ⋆θ)(u) =
Z R
0
Z 2π
0
x(u, r, ϑ)θ(r)drdϑ.
Spectral ﬁlters discussed in Sections 4.4–4.6 fall under this category: they
are based on the Laplacian operator, which is isotropic. Such an approach,
however, discards important directional information, and might fail to extract
edge-like features.
Fixed gauge
An alternative, to which we have already alluded in Sec-
tion 4.4, is to ﬁx some gauge. Monti et al. (2017) used the principal cur-
vature directions: while this choice is not intrinsic and may ambiguous at
ﬂat points (where curvature vanishes) or uniform curvature (such as on a
perfect sphere), the authors showed that it is reasonable for dealing with
deformable human body shapes, which are approximately piecewise-rigid.
Later works, e.g. Melzi et al. (2019), showed reliable intrinsic construction
of gauges on meshes, computed as (intrinsic) gradients of intrinsic func-
tions. While such tangent ﬁelds might have singularities (i.e., vanish at some
points), the overall procedure is very robust to noise and remeshing.
Angular pooling
Another approach, referred to as angular max pooling,
was used by Masci et al. (2015). In this case, the ﬁlter θ(r, ϑ) is anisotropic,
but its matching with the function is performed over all the possible rotations,
which are then aggregated:
(x ⋆θ)(u) =
max
ϑ0∈[0,2π)
Z R
0
Z 2π
0
x(u, r, ϑ)θ(r, ϑ + ϑ0)drdϑ.
Conceptually, this can be visualised as correlating geodesic patches with a
rotating ﬁlter and collecting the strongest responses.
On meshes, the continuous integrals can be discretised using a construction
referred to as patch operators (Masci et al., 2015). In a geodesic patch around
node u, the neighbour nodes Nu,
Typically multi-hop
neighbours are used.
represented in the local polar coordinates as
(ruv, ϑuv), are weighted by a set of weighting functions w1(r, ϑ), . . . , wK(r, ϑ)
(shown in Figure 18 and acting as ‘soft pixels’) and aggregated,
(x ⋆θ)u =
PK
k=1 wk
P
v∈Nu(ruv, ϑuv)xv θk
PK
k=1 wk
P
v∈Nu(ruv, ϑuv)θk


--- Page 92 ---
88
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
(here θ1, . . . , θK are the learnable coeﬃcients of the ﬁlter). Multi-channel
features are treated channel-wise, with a family of appropriate ﬁlters. Masci
et al. (2015); Boscaini et al. (2016a) used pre-deﬁned weighting functions w,
while Monti et al. (2017) further allowed them to be learnable.
Figure 18: Left-to-right: examples of patch operators used in Geodesic CNN
(Masci et al., 2015), Anisotropic CNN (Boscaini et al., 2016b) and MoNet
(Monti et al., 2017), with the level sets of the weighting functions wk(r, ϑ)
shown in red.
Gauge-equivariant ﬁlters
Both isotropic ﬁlters and angular max pooling
lead to features that are invariant to gauge transformations; they transform
according to the trivial representation ρ(g) = 1 (where g ∈SO(2) is a rotation
of the local coordinate frame). This point of view suggests another approach,
proposed by Cohen et al. (2019); de Haan et al. (2020) and discussed in
Section 4.5, where the features computed by the network are associated with
an arbitrary representation ρ of the structure group G (e.g. SO(2) or O(2)
of rotations or rotations+reﬂections of the coordinate frame, respectively).
Tangent vectors transform according to the standard representation ρ(g) = g.
As another example, the feature vector obtained by matching n rotated copies
of the same ﬁlter transforms by cyclic shifts under rotations of the gauge;
this is known as the regular representation of the cyclic group Cn.
As discussed in Section 4.5, when dealing with such geometric features
(associated to a non-trivial representation), we must ﬁrst parallel transport
them to the same vector space before applying the ﬁlter. On a mesh, this can
be implemented via the following message passing mechanism described
by de Haan et al. (2020). Let xu ∈Rd be a d-dimensional input feature at
mesh node u. This feature is expressed relative to an (arbitrary) choice of
gauge at u, and is assumed to transform according to a representation ρin of


--- Page 93 ---
5. GEOMETRIC DEEP LEARNING MODELS
89
G = SO(2) under rotations of the gauge. Similarly, the output features hu of
the mesh convolution are d′ dimensional and should transform according to
ρout (which can be chosen at will by the network designer).
By analogy to Graph Neural Networks, we can implement the gauge-equivariant
convolution (23) on meshes by sending messages from the neighbours Nu
of u (and from u itself) to u:
hu = Θself xu +
X
v∈Nu
Θneigh(ϑuv)ρ(gv→u)xv,
(37)
where Θself, Θneigh(ϑuv) ∈Rd′×d are learned ﬁlter matrices. The structure
group element gv→u ∈SO(2) denotes the eﬀect of parallel transport from v
to u, expressed relative to the gauges at u and v, and can be precomputed
for each mesh. Its action is encoded by a transporter matrix ρ(gv→u) ∈Rd×d.
Note that d is the feature
dimension and is not
necessarily equal to 2, the
dimension of the mesh.
The matrix Θneigh(ϑuv) depends on the angle ϑuv of the neighbour v to
the reference direction (e.g. ﬁrst axis of the frame) at u, so this kernel is
anisotropic: diﬀerent neighbours are treated diﬀerently.
As explained in Section 4.5, for h(u) to be a well-deﬁned geometric quantity, it
should transform as h(u) 7→ρout(g−1(u))h(u) under gauge transformations.
This will be the case when Θselfρin(ϑ) = ρout(ϑ)Θself for all ϑ ∈SO(2),
Here we abuse the notation,
identifying 2D rotations with
angles ϑ.
and Θneigh(ϑuv −ϑ)ρin(ϑ) = ρout(ϑ)Θneigh(ϑuv). Since these constraints
are linear, the space of matrices Θself and matrix-valued functions Θneigh
satisfying these constraints is a linear subspace, and so we can parameterise
them as a linear combination of basis kernels with learnable coeﬃcients:
Θself = P
i αiΘi
self and Θneigh = P
i βiΘi
neigh.
5.7
Recurrent Neural Networks
Our discussion has thus far always assumed the inputs to be solely spatial
across a given domain. However, in many common use cases, the inputs can
also be considered sequential (e.g. video, text or speech). In this case, we
assume that the input consists of arbitrarily many steps, wherein at each step
t we are provided with an input signal, which we represent as X(t) ∈X(Ω(t)).
Whether the domain is
considered static or dynamic
concerns time scales: e.g., a
road network does change
over time (as new roads are
built and old ones are
demolished), but
signiﬁcantly slower
compared to the ﬂow of
traﬃc. Similarly, in social
networks, changes in
engagement (e.g. Twitter
users re-tweeting a tweet)
happen at a much higher
frequency than changes in
the follow graph.
While in general the domain can evolve in time together with the signals
on it, it is typically assumed that the domain is kept ﬁxed across all the
t, i.e. Ω(t) = Ω. Here, we will exclusively focus on this case, but note that


--- Page 94 ---
90
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
exceptions are common. Social networks are an example where one often has
to account for the domain changing through time, as new links are regularly
created as well as erased. The domain in this setting is often referred to as a
dynamic graph (Xu et al., 2020a; Rossi et al., 2020).
Often, the individual X(t) inputs will exhibit useful symmetries and hence
may be nontrivially treated by any of our previously discussed architectures.
Some common examples include: videos (Ωis a ﬁxed grid, and signals are a
sequence of frames); fMRI scans (Ωis a ﬁxed mesh representing the geometry
of the brain cortex, where diﬀerent regions are activated at diﬀerent times as
a response to presented stimuli); and traﬃc ﬂow networks (Ωis a ﬁxed graph
representing the road network, on which e.g. the average traﬃc speed is
recorded at various nodes).
Let us assume an encoder function f(X(t)) providing latent representations
at the level of granularity appropriate for the problem and respectful of the
symmetries of the input domain. As an example
We do not lose generality in
our example; equivalent
analysis can be done e.g. for
node-level outputs on a
spatiotemporal graph; the
only diﬀerence is in the
choice of encoder f (which
will then be a permutation
equivariant GNN).
, consider processing video
frames: that is, at each timestep, we are given a grid-structured input repre-
sented as an n×d matrix X(t), where n is the number of pixels (ﬁxed in time)
and d is the number of input channels (e.g. d = 3 for RGB frames). Further,
we are interested in analysis at the level of entire frames, in which case it
is appropriate to implement f as a translation invariant CNN, outputting a
k-dimensional representation z(t) = f(X(t)) of the frame at time-step t.
We are now left with the task of appropriately summarising a sequence of
vectors z(t) across all the steps. A canonical way to dynamically aggregate this
information in a way that respects the temporal progression of inputs and
also easily allows for online arrival of novel data-points, is using a Recurrent
Neural Network (RNN).
Note that the z(t) vectors can
be seen as points on a
temporal grid: hence,
processing them with a CNN
is also viable in some cases.
Transformers are also
increasingly popular models
for processing generic
sequential inputs.
What we will show here is that RNNs are an interest-
ing geometric architecture to study in their own right, since they implement
a rather unusual type of symmetry over the inputs z(t).
SimpleRNNs
At each step, the recurrent neural network computes an m-
dimensional summary vector h(t) of all the input steps up to and including
t. This (partial) summary is computed conditional on the current step’s
features and the previous step’s summary, through a shared update function,
R : Rk × Rm →Rm, as follows (see Figure 19 for a summary):
h(t) = R(z(t), h(t−1))
(38)


--- Page 95 ---
5. GEOMETRIC DEEP LEARNING MODELS
91
h(0)
R
R
R
R
. . .
z(1)
z(2)
z(3)
z(4)
h(4)
h(3)
h(2)
h(1)
f
f
f
f
X(1)
X(2)
X(3)
X(4)
h(1)
h(2)
h(3)
. . .
Figure 19: Illustration of processing video input with RNNs. Each input
video frame X(t) is processed using a shared function f—e.g. a transla-
tion invariant CNN—into a ﬂat representation z(t). Then the RNN update
function R is iterated across these vectors, iteratively updating a summary
vector h(t) which summarises all the inputs up to and including z(t). The
computation is seeded with an initial summary vector h(0), which may be
either pre-determined or learnable.


--- Page 96 ---
92
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
and, as both z(t) and h(t−1) are ﬂat vector representations, R may be most
easily expressed as a single fully-connected neural network layer (often
known as SimpleRNN
In spite of their name,
SimpleRNNs are remarkably
expressive. For example, it
was shown by Siegelmann
and Sontag (1995) that such
models are Turing-complete,
meaning that they can likely
represent any computation
we may ever be able to
execute on computers.
; see Elman (1990); Jordan (1997)):
h(t) = σ(Wz(t) + Uh(t−1) + b)
(39)
where W ∈Rk×m, U ∈Rm×m and b ∈Rm are learnable parameters, and
σ is an activation function. While this introduces loops in the network’s
computational graph, in practice the network is unrolled for an appropriate
number of steps, allowing for backpropagation through time (Robinson and
Fallside, 1987; Werbos, 1988; Mozer, 1989) to be applied.
The summary vectors may then be appropriately leveraged for the down-
stream task—if a prediction is required at every step of the sequence, then a
shared predictor may be applied to each h(t) individually. For classifying
entire sequences, typically the ﬁnal summary, h(T), is passed to a classiﬁer.
Here, T is the length of the sequence.
Specially, the initial summary vector is usually either set to the zero-vector,
i.e. h(0) = 0, or it is made learnable. Analysing the manner in which the
initial summary vector is set also allows us to deduce an interesting form of
translation equivariance exhibited by RNNs.
Translation equivariance in RNNs
Since we interpret the individual steps
t as discrete time-steps, the input vectors z(t) can be seen as living on a one-
dimensional
Note that this construction is
extendable to grids in higher
dimensions, allowing us to,
e.g., process signals living on
images in a scanline fashion.
Such a construction powered
a popular series of models,
such as the PixelRNN from
van den Oord et al. (2016b).
grid of time-steps. While it might be attractive to attempt ex-
tending our translation equivariance analysis from CNNs here, it cannot be
done in a trivial manner.
To see why, let us assume that we have produced a new sequence z′(t) = z(t+1)
by performing a left-shift of our sequence by one step. It might be tempting to
attempt showing h′(t) = h(t+1), as one expects with translation equivariance;
however, this will not generally hold. Consider t = 1; directly applying and
expanding the update function, we recover the following:
h′(1) = R(z′(1), h(0)) = R(z(2), h(0))
(40)
h(2) = R(z(2), h(1)) = R(z(2), R(z(1), h(0)))
(41)
Hence, unless we can guarantee that h(0) = R(z(1), h(0)), we will not recover
translation equivariance. Similar analysis can then be done for steps t > 1.


--- Page 97 ---
5. GEOMETRIC DEEP LEARNING MODELS
93
Fortunately, with a slight refactoring of how we represent z, and for a suitable
choice of R, it is possible to satisfy the equality above, and hence demonstrate
a setting in which RNNs are equivariant to shifts. Our problem was largely
one of boundary conditions: the equality above includes z(1), which our left-
shift operation destroyed. To abstract this problem away, we will observe
how an RNN processes an appropriately left-padded sequence, ¯z(t), deﬁned
as follows:
¯z(t) =
(
0
t ≤t′
z(t−t′)
t > t′
Such a sequence now allows for left-shifting
Note that equivalent analyses
will arise if we use a diﬀerent
padding vector than 0.
by up to t′ steps without de-
stroying any of the original input elements. Further, note we do not need to
handle right-shifting separately; indeed, equivariance to right shifts naturally
follows from the RNN equations.
We can now again analyse the operation of the RNN over a left-shifted verson
of ¯z(t), which we denote by ¯z′(t) = ¯z(t+1), as we did in Equations 40–41:
h′(1) = R(¯z′(1), h(0)) = R(¯z(2), h(0))
h(2) = R(¯z(2), h(1)) = R(¯z(2), R(¯z(1), h(0))) = R(¯z(2), R(0, h(0)))
where the substitution ¯z(1) = 0 holds as long as t′ ≥1, i.e. as long as any
padding is applied
In a very similar vein, we can
derive equivariance to
left-shifting by s steps as long
as t′ ≥s.
. Now, we can guarantee equivariance to left-shifting by
one step (h′(t) = h(t+1)) as long as h(0) = R(0, h(0)).
Said diﬀerently, h(0) must be chosen to be a ﬁxed point of a function γ(h) =
R(0, h). If the update function R is conveniently chosen, then not only can
we guarantee existence of such ﬁxed points, but we can even directly obtain
them by iterating the application of R until convergence; e.g., as follows:
h0 = 0
hk+1 = γ(hk),
(42)
where the index k refers to the iteration of R in our computation, as opposed
to the the index (t) denoting the time step of the RNN. If we choose R
such that γ is a contraction mapping
Contractions are functions
γ : X →X such that, under
some norm ∥· ∥on X,
applying γ contracts the
distances between points: for
all x, y ∈X, and some
q ∈[0, 1), it holds that
∥γ(x) −γ(y)∥≤q∥x −y∥.
Iterating such a function then
necessarily converges to a
unique ﬁxed point, as a direct
consequence of Banach’s Fixed
Point Theorem (Banach, 1922).
, such an iteration will indeed converge
to a unique ﬁxed point. Accordingly, we can then iterate Equation (42)
until hk+1 = hk, and we can set h(0) = hk. Note that this computation is
equivalent to left-padding the sequence with “suﬃciently many” zero-vectors.
Depth in RNNs
It is also easy to stack multiple RNNs—simply use the
h(t) vectors as an input sequence for a second RNN. This kind of construc-
tion is occasionally called a “deep RNN”, which is potentially misleading.


--- Page 98 ---
94
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
Eﬀectively, due to the repeated application of the recurrent operation, even
a single RNN “layer” has depth equal to the number of input steps.
This often introduces uniquely challenging learning dynamics when opti-
mising RNNs, as each training example induces many gradient updates to
the shared parameters of the update network. Here we will focus on perhaps
the most prominent such issue—that of vanishing and exploding gradients
(Bengio et al., 1994)—which is especially problematic in RNNs, given their
depth and parameter sharing. Further, it has single-handedly spurred some
of the most inﬂuential research on RNNs. For a more detailed overview,
we refer the reader to Pascanu et al. (2013), who have studied the training
dynamics of RNNs in great detail, and exposed these challenges from a
variety of perspectives: analytical, geometrical, and the lens of dynamical
systems.
To illustrate vanishing gradients, consider a SimpleRNN with a sigmoidal
activation function σ
−6
−5
−4
−3
−2
−1
0
1
2
3
4
5
6
0
0.2
0.4
0.6
0.8
1
Examples of such an
activation include the logistic
function, σ(x) =
1
1+exp(−x),
and the hyperbolic tangent,
σ(x) = tanh x. They are
called sigmoidal due to the
distinct S-shape of their plots.
, whose derivative magnitude |σ′| is always between 0
and 1. Multiplying many such values results in gradients that quickly tend
to zero, implying that early steps in the input sequence may not be able to
have inﬂuence in updating the network parameters at all.
For example, consider the next-word prediction task (common in e.g. predic-
tive keyboards), and the input text “Petar is Serbian. He was born on ...[long
paragraph] ...Petar currently lives in
”. Here, predicting the next word
as “Serbia” may only be reasonably concluded by considering the very start
of the paragraph—but gradients have likely vanished by the time they reach
this input step, making learning from such examples very challenging.
Deep feedforward neural networks have also suﬀered from the vanishing
gradient problem, until the invention of the ReLU activation (which has
gradients equal to exactly zero or one—thus ﬁxing the vanishing gradient
problem). However, in RNNs, using ReLUs may easily lead to exploding
gradients, as the output space of the update function is now unbounded,
and gradient descent will update the cell once for every input step, quickly
building up the scale of the updates. Historically, the vanishing gradient
phenomenon was recognised early on as a signiﬁcant obstacle in the use of
recurrent networks. Coping with this problem motivated the development
of more sophisticated RNN layers, which we describe next.


--- Page 99 ---
5. GEOMETRIC DEEP LEARNING MODELS
95
Wc, Uc, bc
Wi, Ui, bi
Wf, Uf, bf
Wo, Uo, bo
z(t)
h(t−1)
×
+
tanh
×
h(t)
×
M
ec(t)
c(t−1)
i(t)
o(t)
f (t)
c(t)
LSTM
Figure 20: The dataﬂow of the long short-term memory (LSTM), with its
components and memory cell (M) clearly highlighted. Based on the current
input z(t), previous summary h(t−1) and previous cell state c(t−1), the LSTM
predicts the updated cell state c(t) and summary h(t).
5.8
Long Short-Term Memory networks
A key invention that signiﬁcantly reduced the eﬀects of vanishing gradients
in RNNs is that of gating mechanisms, which allow the network to selec-
tively overwrite information in a data-driven way. Prominent examples of
these gated RNNs include the Long Short-Term Memory (LSTM; Hochreiter
and Schmidhuber (1997)) and the Gated Recurrent Unit (GRU; Cho et al.
(2014)). Here we will primarily discuss the LSTM—speciﬁcally, the variant
presented by Graves (2013)—in order to illustrate the operations of such
models. Concepts from LSTMs easily carry over to other gated RNNs.
Throughout this section, it will likely be useful to refer to Figure 20, which
illustrates all of the LSTM operations that we will discuss in text.
The LSTM augments the recurrent computation by introducing a memory
cell, which stores cell state vectors, c(t) ∈Rm, that are preserved between
computational steps. The LSTM computes summary vectors, h(t), directly
based on c(t), and c(t) is, in turn, computed using z(t), h(t−1) and c(t−1).
Critically, the cell is not completely overwritten based on z(t) and h(t−1),
which would expose the network to the same issues as the SimpleRNN.
Instead, a certain quantity of the previous cell state may be retained—and


--- Page 100 ---
96
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
the proportion by which this occurs is explicitly learned from data.
Just like in SimpleRNN, we compute features by using a single fully-connected
neural network layer over the current input step and previous summary:
Note that we have set the
activation function to tanh
here; as LSTMs are designed
to ameliorate the vanishing
gradient problem, it is now
appropriate to use a
sigmoidal activation.
ec(t) = tanh(Wcz(t) + Uch(t−1) + bc)
(43)
But, as mentioned, we do not allow all of this vector to enter the cell—hence
why we call it the vector of candidate features, and denote it as ec(t). Instead,
the LSTM directly learns gating vectors, which are real-valued vectors in the
range [0, 1], and decide how much of the signal should be allowed to enter,
exit, and overwrite the memory cell.
Three such gates are computed, all based on z(t) and h(t−1): the input gate i(t),
which computes the proportion of the candidate vector allowed to enter the
cell; the forget gate f(t), which computes the proportion of the previous cell
state to be retained, and the output gate o(t), which computes the proportion
of the new cell state to be used for the ﬁnal summary vector. Typically all of
these gates are also derived using a single fully connected layer, albeit with
the logistic sigmoid activation logistic(x) =
1
1+exp(−x), in order to guarantee
that the outputs are in the [0, 1] range
Note that the three gates are
themselves vectors, i.e.
i(t), f (t), o(t) ∈[0, 1]m. This
allows them to control how
much each of the m
dimensions is allowed
through the gate.
:
i(t) = logistic(Wiz(t) + Uih(t−1) + bi)
(44)
f(t) = logistic(Wfz(t) + Ufh(t−1) + bf)
(45)
o(t) = logistic(Woz(t) + Uoh(t−1) + bo)
(46)
Finally, these gates are appropriately applied to decode the new cell state,
c(t), which is then modulated by the output gate to produce the summary
vector h(t), as follows:
c(t) = i(t) ⊙ec(t) + f(t) ⊙c(t−1)
(47)
h(t) = o(t) ⊙tanh(c(t))
(48)
where ⊙is element-wise vector multiplication. Applied together, Equations
(43)–(48) completely specify the update rule for the LSTM, which now takes
into account the cell vector c(t) as well
This is still compatible with
the RNN update blueprint
from Equation (38); simply
consider the summary vector
to be the concatenation of h(t)
and c(t); sometimes denoted
by h(t)∥c(t).
:
(h(t), c(t)) = R(z(t), (h(t−1), c(t−1)))
Note that, as the values of f(t) are derived from z(t) and h(t−1)—and therefore
directly learnable from data—the LSTM eﬀectively learns how to appropri-
ately forget past experiences. Indeed, the values of f(t) directly appear in


--- Page 101 ---
5. GEOMETRIC DEEP LEARNING MODELS
97
the backpropagation update for all the LSTM parameters (W∗, U∗, b∗), al-
lowing the network to explicitly control, in a data-driven way, the degree of
vanishing for the gradients across the time steps.
Besides tackling the vanishing gradient issue head-on, it turns out that
gated RNNs also unlock a very useful form of invariance to time-warping
transformations, which remains out of reach of SimpleRNNs.
Time warping invariance of gated RNNs
We will start by illustrating, in a
continuous-time setting
We focus on the continuous
setting as it will be easier to
reason about manipulations
of time there.
, what does it mean to warp time, and what is required
of a recurrent model in order to achieve invariance to such transformations.
Our exposition will largely follow the work of Tallec and Ollivier (2018),
that initially described this phenomenon—and indeed, they were among
the ﬁrst to actually study RNNs from the lens of invariances.
Let us assume a continuous time-domain signal z(t), on which we would
like to apply an RNN. To align the RNN’s discrete-time computation of
summary vectors h(t)
We will use h(t) to denote a
continuous signal at time t,
and h(t) to denote a discrete
signal at time-step t.
with an analogue in the continuous domain, h(t), we
will observe its linear Taylor expansion:
h(t + δ) ≈h(t) + δdh(t)
dt
(49)
and, setting δ = 1, we recover a relationship between h(t) and h(t+1), which
is exactly what the RNN update function R (Equation 38) computes. Namely,
the RNN update function satisﬁes the following diﬀerential equation:
dh(t)
dt
= h(t + 1) −h(t) = R(z(t + 1), h(t)) −h(t)
(50)
We would like the RNN to be resilient to the way in which the signal is
sampled (e.g. by changing the time unit of measurement), in order to
account for any imperfections or irregularities therein. Formally, we denote
a time warping
Such warping operations can
be simple, such as time
rescaling; e.g. τ(t) = 0.7t
(displayed above), which, in
a discrete setting, would
amount to new inputs being
received every ∼1.43 steps.
However, it also admits a
wide spectrum of
variably-changing sampling
rates, e.g. sampling may
freely accelerate or decelerate
throughout the time domain.
operation τ : R+ →R+, as any monotonically increasing
diﬀerentiable mapping between times. The notation τ is chosen because
time warping represents an automorphism of time.
Further, we state that a class of models is invariant to time warping if, for any
model of the class and any such τ, there exists another (possibly the same)
model from the class that processes the warped data in the same way as the
original model did in the non-warped case.


--- Page 102 ---
98
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
This is a potentially very useful property. If we have an RNN class capable
of modelling short-term dependencies well, and we can also show that this
class is invariant to time warping, then we know it is possible to train such a
model in a way that will usefully capture long-term dependencies as well (as
they would correspond to a time dilation warping of a signal with short-term
dependencies). As we will shortly see, it is no coincidence that gated RNN
models such as the LSTM were proposed to model long-range dependencies.
Achieving time warping invariance is tightly coupled with presence of gating
mechanisms, such as the input/forget/output gates of LSTMs.
When time gets warped by τ, the signal observed by the RNN at time t
is z(τ(t)) and, to remain invariant to such warpings, it should predict an
equivalently-warped summary function h(τ(t)). Using Taylor expansion
arguments once more, we derive a form of Equation 50 for the warped time,
that the RNN update R should satisfy:
dh(τ(t))
dτ(t)
= R(z(τ(t + 1)), h(τ(t))) −h(τ(t))
(51)
However, the above derivative is computed with respect to the warped time
τ(t), and hence does not take into account the original signal. To make our
model take into account the warping transformation explicitly, we need to
diﬀerentiate the warped summary function with respect to t. Applying the
chain rule, this yields the following diﬀerential equation:
dh(τ(t))
dt
= dh(τ(t))
dτ(t)
dτ(t)
dt
= dτ(t)
dt R(z(τ(t + 1)), h(τ(t))) −dτ(t)
dt h(τ(t))
(52)
and, for our (continuous-time) RNN to remain invariant to any time warping
τ(t), it needs to be able to explicitly represent the derivative dτ(t)
dt , which
is not assumed known upfront! We need to introduce a learnable function
Γ which approximates this derivative. For example, Γ could be a neural
network taking into account z(t + 1) and h(t) and predicting scalar outputs.
Now, remark that, from the point of view of a discrete RNN model under time
warping, its input z(t) will correspond to z(τ(t)), and its summary h(t) will
correspond to h(τ(t)). To obtain the required relationship of h(t) to h(t+1)
in order to remain invariant to time warping, we will use a one-step Taylor
expansion of h(τ(t)):
h(τ(t + δ)) ≈h(τ(t)) + δdh(τ(t))
dt


--- Page 103 ---
5. GEOMETRIC DEEP LEARNING MODELS
99
and, once again, setting δ = 1 and substituting Equation 52, then discretising:
h(t+1) = h(t) + dτ(t)
dt R(z(t+1), h(t)) −dτ(t)
dt h(t)
= dτ(t)
dt R(z(t+1), h(t)) +

1 −dτ(t)
dt

h(t)
Finally, we swap dτ(t)
dt
with the aforementioned learnable function, Γ. This
gives us the required form for our time warping-invariant RNN:
h(t+1) = Γ(z(t+1), h(t))R(z(t+1), h(t)) + (1 −Γ(z(t+1), h(t)))h(t)
(53)
We may quickly deduce that SimpleRNNs (Equation 39) are not time warping
invariant, given that they do not feature the second term in Equation 53.
Instead, they fully overwrite h(t) with R(z(t+1), h(t)), which corresponds to
assuming no time warping at all; dτ(t)
dt
= 1, i.e. τ(t) = t.
Further, our link between continuous-time RNNs and the discrete RNN
based on R rested on the accuracy of the Taylor approximation, which holds
only if the time-warping derivative is not too large, i.e., dτ(t)
dt
≲1. The
intuitive explanation of this is: if our time warping operation ever contracts
time in a way that makes time increments (t →t + 1) large enough that
intermediate data changes are not sampled, the model can never hope to
process time-warped inputs in the same way as original ones—it simply
would not have access to the same information. Conversely, time dilations of
any form (which, in discrete terms, correspond to interspersing the input
time-series with zeroes) are perfectly allowed within our framework.
Combined with our requirement of monotonically increasing τ ( dτ(t)
dt
>
0), we can bound the output space of Γ as 0 < Γ(z(t+1), h(t)) < 1, which
motivates the use of the logistic sigmoid activation for Γ, e.g.:
Γ(z(t+1), h(t)) = logistic(WΓz(t+1) + UΓh(t) + bΓ)
exactly matching the LSTM gating equations (e.g. Equation 44). The main
diﬀerence is that LSTMs compute gating vectors, whereas Equation 53 implies
Γ should output a scalar. Vectorised gates (Hochreiter, 1991) allow to ﬁt a
diﬀerent warping derivative in every dimension of h(t), allowing for reasoning
over multiple time horizons simultaneously.
It is worth taking a pause here to summarise what we have done. By requiring
that our RNN class is invariant to (non-destructive) time warping, we have


--- Page 104 ---
100
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
derived the necessary form that it must have (Equation 53), and showed that
it exactly corresponds to the class of gated RNNs. The gates’ primary role
under this perspective is to accurately ﬁt the derivative dτ(t)
dt
of the warping
transformation.
The notion of class invariance is somewhat distinct from the invariances
we studied previously. Namely, once we train a gated RNN on a time-
warped input with τ1(t), we typically cannot zero-shot transfer
One case where zero-shot
transfer is possible is when
the second time warping is
assumed to be a time rescaling
of the ﬁrst one
(τ2(t) = ατ1(t)).
Transferring a gated RNN
pre-trained on τ1 to a signal
warped by τ2 merely requires
rescaling the gates:
Γ2(z(t+1), h(t)) =
αΓ1(z(t+1), h(t)). R can
retain its parameters
(R1 = R2).
it to a signal
warped by a diﬀerent τ2(t). Rather, class invariance only guarantees that
gated RNNs are powerful enough to ﬁt both of these signals in the same
manner, but potentially with vastly diﬀerent model parameters. That being
said, the realisation that eﬀective gating mechanisms are tightly related to
ﬁtting the warping derivative can yield useful prescriptions for gated RNN
optimisation, as we now brieﬂy demonstrate.
For example, we can often assume that the range of the dependencies we are
interested in tracking within our signal will be in the range [Tl, Th] time-steps.
By analysing the analytic solutions to Equation 52, it can be shown that
the characteristic forgetting time of h(t) by our gated RNN is proportional to
1
Γ(z(t+1),h(t)). Hence, we would like our gating values to lie between
h
1
Th ,
1
Tm
i
in order to eﬀectively remember information within the assumed range.
Further, if we assume that z(t) and h(t) are roughly zero-centered—which is
a common by-product of applying transformations such as layer normali-
sation (Ba et al., 2016)—we can assume that E[Γ(z(t+1), h(t))] ≈logistic(bΓ).
Controlling the bias vector of the gating mechanism is hence a very powerful
way of controlling the eﬀective gate value
This insight was already
spotted by Gers and
Schmidhuber (2000);
Jozefowicz et al. (2015), who
empirically recommended
initialising the forget-gate
bias of LSTMs to a constant
positive vector, such as 1.
.
Combining the two observations, we conclude that an appropriate range of
gating values can be obtained by initialising bΓ ∼−log(U(Tl, Th)−1), where
U is the uniform real distribution. Such a recommendation was dubbed
chrono initialisation by Tallec and Ollivier (2018), and has been empirically
shown to improve the long-range dependency modelling of gated RNNs.
Sequence-to-sequence learning with RNNs
One prominent historical ex-
ample of using RNN-backed computation are sequence-to-sequence translation
tasks, such as machine translation of natural languages. The pioneering seq2seq
work by Sutskever et al. (2014) achieved this by passing the summary vector,


--- Page 105 ---
5. GEOMETRIC DEEP LEARNING MODELS
101
h(0)
Renc
Renc
Renc
Renc
z(1)
z(2)
z(3)
z(4)
h(4)
Rdec
Rdec
Rdec
. . .
y(0)
y(1)
y(2)
. . .
y(1)
y(2)
y(3)
h(1)
h(2)
h(3)
eh(1)
eh(2)
eh(3)
Figure 21: One typical example of a seq2seq architecture with an RNN
encoder Renc and RNN decoder Rdec. The decoder is seeded with the ﬁnal
summary vector h(T) coming out of the encoder, and then proceeds in an
autoregressive fashion: at each step, the predicted output from the previous
step is fed back as input to Rdec. The bottleneck problem is also illustrated
with the red lines: the summary vector h(T) is pressured to store all relevant
information for translating the input sequence, which becomes increasingly
challenging as the input length grows.
h(T) as an initial input for a decoder RNN, with outputs of RNN blocks being
given as inputs for the next step.
This placed substantial representational pressure on the summary vector,
h(T). Within the context of deep learning, h(T) is sometimes referred to as a
bottleneck
The bottleneck eﬀect has
recently received substantial
attention in the graph
representation learning
community (Alon and Yahav,
2020), as well as neural
algorithmic reasoning
(Cappart et al., 2021).
. Its ﬁxed capacity must be suﬃcient for representing the content of
the entire input sequence, in a manner that is conducive to generating a cor-
responding sequence, while also supporting input sequences of substantially
diﬀerent lengths (Figure 21).
In reality, diﬀerent steps of the output may wish to focus (attend) on diﬀerent
parts of the input, and all such choices are diﬃcult to represent via a bottle-
neck vector. Following from this observation, the popular recurrent attention
model was proposed by Bahdanau et al. (2014). At every step of processing,
a query vector is generated by an RNN; this query vector then interacts with
the representation of every time-step h(t), primarily by computing a weighted
sum over them. This model pioneered neural content-based attention and
predates the success of the Transformer model.
Lastly, while attending oﬀers a soft way to dynamically focus on parts of
