# 4 Geometric Domains: the 5 Gs

--- Page 34 ---
30
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
(grid). Graph classiﬁcation is an example of the latter setting, where both
the structure of the graph as well as the signal deﬁned on it (e.g. node
features) are important. In the case of varying domain, geometric stability
(in the sense of insensitivity to the deformation of Ω) plays a crucial role in
Geometric Deep Learning architectures.
This blueprint has the right level of generality to be used across a wide
range of geometric domains. Diﬀerent Geometric Deep Learning methods
thus diﬀer in their choice of the domain, symmetry group, and the speciﬁc
implementation details of the aforementioned building blocks. As we will
see in the following, a large class of deep learning architectures currently in
use fall into this scheme and can thus be derived from common geometric
principles.
In the following sections (4.1–4.6) we will describe the various geometric
domains focusing on the ‘5G’, and in Sections 5.1–5.8 the speciﬁc implemen-
tations of Geometric Deep Learning on these domains.
Architecture
Domain Ω
Symmetry group G
CNN
Grid
Translation
Spherical CNN
Sphere / SO(3)
Rotation SO(3)
Intrinsic / Mesh CNN
Manifold
Isometry Iso(Ω) /
Gauge symmetry SO(2)
GNN
Graph
Permutation Σn
Deep Sets
Set
Permutation Σn
Transformer
Complete Graph
Permutation Σn
LSTM
1D Grid
Time warping
4
Geometric Domains: the 5 Gs
The main focus of our text will be on graphs, grids, groups, geodesics, and
gauges. In this context, by ‘groups’ we mean global symmetry transforma-
tions in homogeneous space, by ‘geodesics’ metric structures on manifolds,
and by ‘gauges’ local reference frames deﬁned on tangent bundles (and vec-


--- Page 35 ---
4. GEOMETRIC DOMAINS: THE 5 GS
31
Figure 9: The 5G of Geometric Deep Learning: grids, groups & homogeneous
spaces with global symmetry, graphs, geodesics & metrics on manifolds,
and gauges (frames for tangent or feature spaces).
tor bundles in general). These notions will be explained in more detail later.
In the next sections, we will discuss in detail the main elements in common
and the key distinguishing features between these structures and describe
the symmetry groups associated with them. Our exposition is not in the
order of generality – in fact, grids are particular cases of graphs – but a way
to highlight important concepts underlying our Geometric Deep Learning
blueprint.
4.1
Graphs and Sets
In multiple branches of science, from sociology to particle physics, graphs
are used as models of systems of relations and interactions. From our per-
spective, graphs give rise to a very basic type of invariance modelled by the
group of permutations. Furthermore, other objects of interest to us, such as
grids and sets, can be obtained as a particular case of graphs.
A graph G = (V, E) is a collection of nodes
Depending on the
application ﬁeld, nodes may
also be called vertices, and
edges are often referred to as
links or relations. We will use
these terms interchangeably.
V and edges E ⊆V×V between pairs
of nodes. For the purpose of the following discussion, we will further assume
the nodes to be endowed with s-dimensional node features, denoted by xu for
all u ∈V. Social networks are perhaps among the most commonly studied
examples of graphs, where nodes represent users, edges correspond to
friendship relations between them, and node features model user properties
such as age, proﬁle picture, etc. It is also often possible to endow the edges,
or entire graphs, with features;
Isomorphism is an
edge-preserving bijection
between two graphs. Two
isomorphic graphs shown
here are identical up to
reordering of their nodes.
but as this does not alter the main ﬁndings
of this section, we will defer discussing it to future work.


--- Page 36 ---
32
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
The key structural property of graphs is that the nodes in V are usually not
assumed to be provided in any particular order, and thus any operations
performed on graphs should not depend on the ordering of nodes. The
desirable property that functions acting on graphs should satisfy is thus
permutation invariance, and it implies that for any two isomorphic graphs, the
outcomes of these functions are identical. We can see this as a particular
setting of our blueprint, where the domain Ω= G and the space X(G, Rd)
is that of d-dimensional node-wise signals. The symmetry we consider is
given by the permutation group G = Σn, whose elements are all the possible
orderings of the set of node indices {1, . . . , n}.
Let us ﬁrst illustrate the concept of permutation invariance on sets, a special
case of graphs without edges (i.e., E = ∅). By stacking the node features
as rows of the n × d matrix X = (x1, . . . , xn)⊤, we do eﬀectively specify an
ordering of the nodes. The action of the permutation g ∈Σn on the set of
nodes amounts to the reordering of the rows of X, which can be represented
as an n×n permutation matrix ρ(g) = P,
There are exactly n! such
permutations, so Σn is, even
for modest n, a very large
group.
where each row and column contains
exactly one 1 and all the other entries are zeros.
A function f operating on this set is then said to be permutation invariant if,
for any such permutation matrix P, it holds that f(PX) = f(X). One simple
such function is
f(X) = φ
 X
u∈V
ψ (xu)
!
,
(6)
where the function ψ is independently applied to every node’s features, and
φ is applied on its sum-aggregated outputs: as sum is independent of the order
in which its inputs are provided, such a function is invariant with respect to
the permutation of the node set, and is hence guaranteed to always return
the same output, no matter how the nodes are permuted.
Functions like the above provide a ‘global’ graph-wise output, but very often,
we will be interested in functions that act ‘locally’, in a node-wise manner.
For example, we may want to apply some function to update the features in
every node, obtaining the set of latent node features. If we stack these latent
features into a matrix H = F(X)
We use the bold notation for
our function F(X) to
emphasise it outputs
node-wise vector features
and is hence a matrix-valued
function.
is no longer permutation invariant: the
order of the rows of H should be tied to the order of the rows of X, so that
we know which output node feature corresponds to which input node. We
need instead a more ﬁne-grained notion of permutation equivariance, stating
that, once we “commit” to a permutation of inputs, it consistently permutes
the resulting objects. Formally, F(X) is a permutation equivariant function


--- Page 37 ---
4. GEOMETRIC DOMAINS: THE 5 GS
33
if, for any permutation matrix P, it holds that F(PX) = PF(X). A shared
node-wise linear transform
FΘ(X) = XΘ
(7)
speciﬁed by a weight matrix Θ ∈Rd×d′, is one possible construction of such a
permutation equivariant function, producing in our example latent features
of the form hu = Θ⊤xu.
This construction arises naturally from our Geometric Deep Learning blueprint.
We can ﬁrst attempt to characterise linear equivariants (functions of the form
FPX = PFX), for which it is easy to verify that any such map can be writ-
ten as a linear combination of two generators, the identity F1X = X and the
average F2X = 1
n11⊤X = 1
n
Pn
u=1 xu. As will be described in Section 5.4,
the popular Deep Sets (Zaheer et al., 2017) architecture follows precisely
this blueprint.
We can now generalise the notions of permutation invariance and equivari-
ance from sets to graphs. In the generic setting E ̸= ∅, the graph connectivity
can be represented by the n × n adjacency matrix A,
When the graph is undirected,
i.e. (u, v) ∈E iﬀ(v, u) ∈E,
the adjacency matrix is
symmetric, A = A⊤.
deﬁned as
auv =
(
1
(u, v) ∈E
0
otherwise.
(8)
Note that now the adjacency and feature matrices A and X are “synchro-
nised”, in the sense that auv speciﬁes the adjacency information between
the nodes described by the uth and vth rows of X. Therefore, applying a
permutation matrix P to the node features X automatically implies applying
it to A’s rows and columns, PAP⊤.
PAP⊤is the representation
of Σn acting on matrices.
We say that (a graph-wise function) f
is permutation invariant if
f(PX, PAP⊤) = f(X, A)
As a way to emphasise the
fact that our functions
operating over graphs now
need to take into account the
adjacency information, we
use the notation f(X, A).
(9)
and (a node-wise function) F is permutation equivariant if
F(PX, PAP⊤) = PF(X, A)
(10)
for any permutation matrix P.
Here again, we can ﬁrst characterise linear equivariant functions.
This corresponds to the Bell
number B4, which counts the
number of ways to partition a
set of 4 elements, in this case
given by the 4-indices
(u, v), (u′, v′) indexing a
linear map acting on the
adjacency matrix.
As ob-
served by Maron et al. (2018), any linear F satisfying equation (10) can be
expressed as a linear combination of ﬁfteen linear generators; remarkably,


--- Page 38 ---
34
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
this family of generators is independent of n. Amongst these generators, our
blueprint speciﬁcally advocates for those that are also local, i.e., whereby the
output on node u directly depends on its neighbouring nodes in the graph.
We can formalise this constraint explicitly in our model construction, by
deﬁning what it means for a node to be neighbouring another.
A (undirected) neighbourhood of node u, sometimes also called 1-hop, is
deﬁned as
Often, the node u itself is
included in its own
neighbourhood.
Nu = {v : (u, v) ∈E or (v, u) ∈E}
(11)
and the neighbourhood features as the multiset
XNu = {{xv : v ∈Nu}}.
A multiset, denoted {{ . . . }}, is
a set where the same element
can appear more than once.
This is the case here because
the features of diﬀerent
nodes can be equal.
(12)
Operating on 1-hop neighbourhoods aligns well with the locality aspect of
our blueprint: namely, deﬁning our metric over graphs as the shortest path
distance between nodes using edges in E.
The GDL blueprint thus yields a general recipe for constructing permutation
equivariant functions on graphs, by specifying a local function φ that operates
over the features of a node and its neighbourhood, φ(xu, XNu). Then, a
permutation equivariant function F can be constructed by applying φ to
every node’s neighbourhood in isolation (see Figure 10):
F(X, A) =


φ(x1, XN1)
φ(x2, XN2)
...
φ(xn, XNn)


(13)
As F is constructed by applying a shared function φ to each node locally,
its permutation equivariance rests on φ’s output being independent on the
ordering of the nodes in Nu. Thus, if φ is built to be permutation invariant,
then this property is satisﬁed. As we will see in future work, the choice of
φ plays a crucial role in the expressive power of such a scheme. When φ is
injective, it is equivalent to one step of the Weisfeiler-Lehman graph isomorphism
test, a classical algorithm in graph theory providing a necessary condition
for two graphs to be isomorphic by an iterative color reﬁnement procedure.
It is also worth noticing that the diﬀerence between functions deﬁned on sets
and more general graphs in this example is that in the latter case we need to
explicitly account for the structure of the domain. As a consequence, graphs
stand apart in the sense that the domain becomes part of the input in machine


--- Page 39 ---
4. GEOMETRIC DOMAINS: THE 5 GS
35
xb
xa
xc
xd
xe
hb
φ(xb, XNb)
Figure 10: An illustration of constructing permutation-equivariant func-
tions over graphs, by applying a permutation-invariant function φ to every
neighbourhood. In this case, φ is applied to the features xb of node b as well
as the multiset of its neighbourhood features, XNb = {{xa, xb, xc, xd, xe}}.
Applying φ in this manner to every node’s neighbourhood recovers the rows
of the resulting matrix of latents features H = F(X, A).
learning problems, whereas when dealing with sets and grids (both particu-
lar cases of graphs) we can specify only the features and assume the domain
to be ﬁxed. This distinction will be a recurring motif in our discussion. As a re-
sult, the notion of geometric stability (invariance to domain deformation) is
crucial in most problems of learning on graphs. It straightforwardly follows
from our construction that permutation invariant and equivariant functions
produce identical outputs on isomorphic (topologically-equivalent) graphs.
These results can be generalised to approximately isomorphic graphs, and
several results on stability under graph perturbations exist (Levie et al.,
2018). We will return to this important point in our discussion on manifolds,
which we will use as an vehicle to study such invariance in further detail.
Second, due to their additional structure, graphs and grids, unlike sets,
can be coarsened in a non-trivial way
More precisely, we cannot
deﬁne a non-trivial
coarsening assuming set
structure alone. There exist
established approaches that
infer topological structure
from unordered sets, and
those can admit non-trivial
coarsening.
, giving rise to a variety of pooling
operations.
4.2
Grids and Euclidean spaces
The second type of objects we consider are grids. It is fair to say that the
impact of deep learning was particularly dramatic in computer vision, natural
language processing, and speech recognition. These applications all share a
geometric common denominator: an underlying grid structure. As already


--- Page 40 ---
36
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
mentioned, grids are a particular case of graphs with special adjacency.
However, since the order of nodes in a grid is ﬁxed, machine learning models
for signals deﬁned on grids are no longer required to account for permutation
invariance, and have a stronger geometric prior: translation invariance.
Circulant matrices and Convolutions
Let us dwell on this point in more
detail. Assuming for simplicity periodic boundary conditions, we can think
of a one-dimensional grid as a ring graph with nodes indexed by 0, 1, . . . , n−1
modulo n (which we will omit for notation brevity) and the adjacency
matrix with elements au,u+1 mod n = 1 and zero otherwise. There are two
main diﬀerences from the general graph case we have discussed before.
First, each node u has identical connectivity, to its neighbours u −1 and
u + 1, and thus structure-wise indistinguishable from the others.
As we will see later, this
makes the grid a
homogeneous space.
Second
and more importantly, since the nodes of the grid have a ﬁxed ordering,
we also have a ﬁxed ordering of the neighbours: we can call u −1 the ‘left
neighbour’ and u + 1 the ‘right neighbour’. If we use our previous recipe
for designing a equivariant function F using a local aggregation function
φ, we now have f(xu) = φ(xu−1, xu, xu+1) at every node of the grid: φ does
not need to be permutation invariant anymore. For a particular choice of a
linear transformation φ(xu−1, xu, xu+1) = θ−1xu−1 + θ0xu + θ1xu+1, we can
write F(X) as a matrix product,
F(X) =


θ0
θ1
θ−1
θ−1
θ0
θ1
...
...
...
θ−1
θ0
θ1
θ1
θ−1
θ0




x0
x1
...
xn−2
xn−1


Note this very special multi-diagonal structure with one element repeated
along each diagonal, sometimes referred to as “weight sharing” in the ma-
chine learning literature.
More generally, given a vector θ = (θ0, . . . , θn−1), a circulant matrix C(θ) =
(θu−v mod n) is obtained by appending circularly shifted versions of the vector
θ. Circulant matrices are synonymous with discrete convolutions,
Because of the periodic
boundary conditions, it is a
circular or cyclic convolution.
In signal processing, θ is
often referred to as the “ﬁlter,”
and in CNNs, its coeﬃcients
are learnable.
(x ⋆θ)u =
n−1
X
v=0
xv mod n θu−v mod n


--- Page 41 ---
4. GEOMETRIC DOMAINS: THE 5 GS
37
as one has C(θ)x = x ⋆θ. A particular choice of θ = (0, 1, 0, . . . , 0)⊤yields a
special circulant matrix that shifts vectors to the right by one position. This
matrix is called the (right) shift or translation operator and denoted by S.
The left shift operator is
given by S⊤. Obviously,
shifting left and then right
(or vice versa) does not do
anything, which means S is
orthogonal: S⊤S = SS⊤= I.
Circulant matrices can be characterised by their commutativity property: the
product of circulant matrices is commutative, i.e. C(θ)C(η) = C(η)C(θ)
for any θ and η. Since the shift is a circulant matrix, we get the familiar
translation or shift equivariance of the convolution operator,
SC(θ)x = C(θ)Sx.
Such commutativity property should not be surprising, since the underlying
symmetry group (the translation group) is Abelian. Moreover, the opposite
direction appears to be true as well, i.e. a matrix is circulant iﬀit commutes
with shift. This, in turn, allows us to deﬁne convolution as a translation equiv-
ariant linear operation, and is a nice illustration of the power of geometric
priors and the overall philosophy of Geometric ML: convolution emerges
from the ﬁrst principle of translational symmetry.
Note that unlike the situation on sets and graphs, the number of linearly
independent shift-equivariant functions (convolutions) grows with the size
of the domain (since we have one degree of freedom in each diagonal of a
circulant matrix). However, the scale separation prior guarantees ﬁlters can
be local, resulting in the same Θ(1)-parameter complexity per layer, as we
will verify in Section 5.1 when discussing the use of these principles in the
implementation of Convolutional Neural Network architectures.
Derivation of the discrete Fourier transform
We have already mentioned
the Fourier transform and its connection to convolution: the fact that the
Fourier transform diagonalises the convolution operation is an important
property used in signal processing to perform convolution in the frequency
domain as an element-wise product of the Fourier transforms. However,
textbooks usually only state this fact, rarely explaining where the Fourier
transform comes from and what is so special about the Fourier basis. Here
we can show it, demonstrating once more how foundational are the basic
principles of symmetry.
For this purpose, recall a fact from linear
We must additionally assume
distinct eigenvalues,
otherwise there might be
multiple possible
diagonalisations. This
assumption is satisﬁed with
our choice of S.
algebra that (diagonalisable) ma-
trices are joinly diagonalisable iﬀthey mutually commute. In other words,
there exists a common eigenbasis for all the circulant matrices, in which


--- Page 42 ---
38
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
they diﬀer only by their eigenvalues. We can therefore pick one circulant
matrix and compute its eigenvectors—we are assured that these will be the
eigenvectors of all other circulant matrices as well. It is convenient to pick the
shift operator, for which the eigenvectors happen to be the discrete Fourier
basis
S is orthogonal but
non-symmetric, hence, its
eigenvectors are orthogonal
but the eigenvalues are
complex (roots of unity).
ϕk =
1
√n

1, e
2πik
n , e
4πik
n , . . . , e
2πi(n−1)k
n
⊤
,
k = 0, 1, . . . , n −1,
which we can arrange into an n × n Fourier matrix Φ = (ϕ0, . . . , ϕn−1).
Multiplication by Φ∗
Note that the eigenvectors are
complex, so we need to take
complex conjugation when
transposing Φ.
gives the Discrete Fourier Transform (DFT), and by Φ
the inverse DFT,
ˆxk =
1
√n
n−1
X
u=0
xue−2πiku
n
xu =
1
√n
n−1
X
k=0
ˆxke+ 2πiku
n
.
Since all circulant matrices are jointly diagonalisable,
Since the Fourier transform is
an orthogonal matrix
(Φ∗Φ = I), geometrically it
acts as a change of the system
of coordinates that amounts
to an n-dimensional rotation.
In this system of coordinates
(“Fourier domain”), the
action of a circulant C matrix
becomes element-wise
product.
they are also diago-
nalised by the Fourier transform and diﬀer only in their eigenvalues. Since
the eigenvalues of the circulant matrix C(θ) are the Fourier transform of
the ﬁlter (see e.g. Bamieh (2018)), ˆθ = Φ∗θ, we obtain the Convolution
Theorem:
C(θ)x = Φ


ˆθ0
...
ˆθn−1

Φ∗x = Φ(ˆθ ⊙ˆx)
Because the Fourier matrix Φ has a special algebraic structure, the prod-
ucts Φ⋆x and Φx can be computed with O(n log n) complexity using a Fast
Fourier Transform (FFT) algorithm. This is one of the reasons why frequency-
domain ﬁltering is so popular in signal processing; furthermore, the ﬁlter is
typically designed directly in the frequency domain, so the Fourier transform
ˆθ is never explicitly computed.
Besides the didactic value of the derivation of the Fourier transform and
convolution we have done here, it provides a scheme to generalise these
concepts to graphs. Realising that the adjacency matrix of the ring graph is
exactly the shift operator, one can can develop the graph Fourier transform
and an analogy of the convolution operator by computing the eigenvectors
of the adjacency matrix (see e.g. Sandryhaila and Moura (2013)). Early


--- Page 43 ---
4. GEOMETRIC DOMAINS: THE 5 GS
39
attempts to develop graph neural networks by analogy to CNNs, sometimes
termed ‘spectral GNNs’, exploited this exact blueprint.
In graph signal processing,
the eigenvectors of the graph
Laplacian are often used as an
alternative of the adjacency
matrix to construct the graph
Fourier transform, see
Shuman et al. (2013). On
grids, both matrices have
joint eigenvectors, but on
graphs they results in
somewhat diﬀerent though
related constructions.
We will see in Sec-
tions 4.4–4.6 that this analogy has some important limitations. The ﬁrst
limitation comes from the fact that a grid is ﬁxed, and hence all signals on it
can be represented in the same Fourier basis. In contrast, on general graphs,
the Fourier basis depends on the structure of the graph. Hence, we cannot
directly compare Fourier transforms on two diﬀerent graphs — a problem
that translated into a lack of generalisation in machine learning problems.
Secondly, multi-dimensional grids, which are constructed as tensor products
of one-dimensional grids, retain the underlying structure: the Fourier basis
elements and the corresponding frequencies (eigenvalues) can be organised
in multiple dimensions. In images, for example, we can naturally talk about
horizontal and vertical frequency and ﬁlters have a notion of direction. On
graphs, the structure of the Fourier domain is one-dimensional, as we can
only organise the Fourier basis functions by the magnitude of the corre-
sponding frequencies. As a result, graph ﬁlters are oblivious of direction or
isotropic.
Derivation of the continuous Fourier transform
For the sake of complete-
ness, and as a segway for the next discussion, we repeat our analysis in the
continuous setting. Like in Section 3.4, consider functions deﬁned on Ω= R
and the translation operator (Svf)(u) = f(u −v) shifting f by some posi-
tion v. Applying Sv to the Fourier basis functions ϕξ(u) = eiξu yields, by
associativity of the exponent,
Sveiξu = eiξ(u−v) = e−iξveiξu,
i.e., ϕuξ(u) is the complex eigenvector of Sv with the complex eigenvalue
e−iξv – exactly mirroring the situation we had in the discrete setting. Since
Sv is a unitary operator (i.e., ∥Svx∥p = ∥x∥p for any p and x ∈Lp(R)),
any eigenvalue λ must satisfy |λ| = 1, which corresponds precisely to the
eigenvalues e−iξv found above. Moreover, the spectrum of the translation
operator is simple, meaning that two functions sharing the same eigenvalue
must necessarily be collinear. Indeed, suppose that Svf = e−iξ0vf for some
ξ0. Taking the Fourier transform in both sides, we obtain
∀ξ , e−iξv ˆf(ξ) = e−iξ0v ˆf(ξ) ,
which implies that ˆf(ξ) = 0 for ξ ̸= ξ0, thus f = αϕξ0.


--- Page 44 ---
40
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
For a general linear operator C that is translation equivariant (SvC = CSv),
we have
SvCeiξu = CSveiξu = e−iξvCeiξu,
implying that Ceiξu is also an eigenfunction
Eigenfunction is synonymous
with ‘eigenvector’ and is used
when referring to
eigenvectors of continuous
operators.
of Sv with eigenvalue e−iξv, from
where it follows from the simplicity of spectrum that Ceiξu = βϕξ(u); in
other words, the Fourier basis is the eigenbasis of all translation equivari-
ant operators. As a result, C is diagonal in the Fourier domain and can be
expressed as Ceiξu = ˆpC(ξ)eiξu, where ˆpC(ξ) is a transfer function acting on
diﬀerent frequencies ξ. Finally, for an arbitrary function x(u), by linearity,
(Cx)(u)
=
C
Z +∞
−∞
ˆx(ξ)eiξudξ =
Z +∞
−∞
ˆx(ξ)ˆpC(ξ)eiξudξ
=
Z +∞
−∞
pC(v)x(u −v)dv = (x ⋆pC)(u),
The spectral characterisation
of the translation group is a
particular case of a more
general result in Functional
Analysis, the Stone’s Theorem,
which derives an equivalent
characterisation for any
one-parameter unitary group.
where pC(u) is the inverse Fourier transform of ˆpC(ξ). It thus follows that
every linear translation equivariant operator is a convolution.
4.3
Groups and Homogeneous spaces
Our discussion of grids highlighted how shifts and convolutions are inti-
mately connected: convolutions are linear shift-equivariant
Technically, we need the
group to be locally compact, so
that there exists a
left-invariant Haar measure.
Integrating with respect to
this measure, we can “shift”
the integrand by any group
element and obtain the same
result, just as how we have
Z +∞
−∞
x(u)du =
Z +∞
−∞
x(u−v)du
for functions x : R →R.
operations, and
vice versa, any shift-equivariant linear operator is a convolution. Further-
more, shift operators can be jointly diagonalised by the Fourier transform.
As it turns out, this is part of a far larger story: both convolution and the
Fourier transform can be deﬁned for any group of symmetries that we can sum
or integrate over.
Consider the Euclidean domain Ω= R. We can understand the convolution
as a pattern matching operation: we match shifted copies of a ﬁlter θ(u) with
an input signal x(u). The value of the convolution (x ⋆θ)(u) at a point u is
the inner product of the signal x with the ﬁlter shifted by u,
(x ⋆θ)(u) = ⟨x, Suθ⟩=
Z
R
x(v)θ(u + v)dv.
Note that what we deﬁne
here is not convolution but
cross-correlation, which is
tacitly used in deep learning
under the name ‘convolution’.
We do it for consistency with
the following discussion,
since in our notation
(ρ(g)x)(u) = x(u −v) and
(ρ(g−1)x)(u) = x(u + v).
Note that in this case u is both a point on the domain Ω= R and also an element
of the translation group, which we can identify with the domain itself, G = R.
We will now show how to generalise this construction, by simply replacing
the translation group by another group G acting on Ω.


--- Page 45 ---
4. GEOMETRIC DOMAINS: THE 5 GS
41
Group convolution
As discussed in Section 3, the action of the group G
on the domain Ωinduces a representation ρ of G on the space of signals
X(Ω) via ρ(g)x(u) = x(g−1u). In the above example, G is the translation
group whose elements act by shifting the coordinates, u + v, whereas ρ(g) is
the shift operator acting on signals as (Svx)(u) = x(u −v). Finally, in order
to apply a ﬁlter to the signal, we invoke our assumption of X(Ω) being a
Hilbert space, with an inner product
⟨x, θ⟩=
Z
Ω
x(u)θ(u)du,
The integration is done w.r.t.
an invariant measure µ on Ω.
In case µ is discrete, this
means summing over Ω.
where we assumed, for the sake of simplicity, scalar-valued signals, X(Ω, R);
in general the inner product has the form of equation (2).
Having thus deﬁned how to transform signals and match them with ﬁlters,
we can deﬁne the group convolution for signals on Ω,
(x ⋆θ)(g) = ⟨x, ρ(g)θ⟩=
Z
Ω
x(u)θ(g−1u)du.
(14)
Note that x⋆θ takes values on the elements g of our group G rather than points
on the domain Ω. Hence, the next layer, which takes x ⋆θ as input, should
act on signals deﬁned on to the group G, a point we will return to shortly.
Just like how the traditional Euclidean convolution is shift-equivariant, the
more general group convolution is G-equivariant. The key observation is
that matching the signal x with a g-transformed ﬁlter ρ(g)θ is the same as
matching the inverse transformed signal ρ(g−1)x with the untransformed
ﬁlter θ. Mathematically, this can be expressed as ⟨x, ρ(g)θ⟩= ⟨ρ(g−1)x, θ⟩.
With this insight, G-equivariance of the group convolution (14) follows
immediately from its deﬁnition and the deﬁning property ρ(h−1)ρ(g) =
ρ(h−1g) of group representations,
(ρ(h)x ⋆θ)(g) = ⟨ρ(h)x, ρ(g)θ⟩= ⟨x, ρ(h−1g)θ⟩= ρ(h)(x ⋆θ)(g).
Let us look at some examples. The case of one-dimensional grid we have
studied above is obtained with the choice Ω= Zn = {0, . . . , n −1} and
the cyclic shift group G = Zn. The group elements in this case are cyclic
shifts of indices, i.e., an element g ∈G can be identiﬁed with some u =
0, . . . , n −1 such that g.v = v −u mod n, whereas the inverse element is
g−1.v = v + u mod n. Importantly, in this example the elements of the group
(shifts) are also elements of the domain (indices). We thus can, with some


--- Page 46 ---
42
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
abuse of notation, identify the two structures (i.e., Ω= G); our expression
for the group convolution in this case
(x ⋆θ)(g) =
n−1
X
v=0
xv θg−1v,
leads to the familiar convolution
Actually here again, this is
cross-correlation.
(x ⋆θ)u =
n−1
X
v=0
xv θv+u mod n.
Spherical convolution
Now consider
Cosmic microwave
background radiation,
captured by the Planck space
observatory, is a signal on S2.
the two-dimensional sphere Ω= S2
with the group of rotations, the special orthogonal group G = SO(3). While
chosen for pedagogical reason, this example is actually very practical and
arises in numerous applications. In astrophysics, for example, observational
data often naturally has spherical geometry. Furthermore, spherical sym-
metries are very important in applications in chemistry when modeling
molecules and trying to predict their properties, e.g. for the purpose of
virtual drug screening.
Representing a point on the sphere as a three-dimensional unit vector u :
∥u∥= 1, the action of the group can be represented as a 3 × 3 orthogonal
matrix R with det(R) = 1. The spherical convolution can thus be written as
the inner product between the signal and the rotated ﬁlter,
(x ⋆θ)(R) =
Z
S2 x(u)θ(R−1u)du.
The action of SO(3) group on
S2. Note that three types of
rotation are possible; the
SO(3) is a three-dimensional
manifold.
The ﬁrst thing to note is than now the group is not identical to the domain:
the group SO(3) is a Lie group that is in fact a three-dimensional manifold,
whereas S2 is a two-dimensional one. Consequently, in this case, unlike the
previous example, the convolution is a function on SO(3) rather than on Ω.
This has important practical consequences: in our Geometric Deep Learn-
ing blueprint, we concatenate multiple equivariant maps (“layers” in deep
learning jargon) by applying a subsequent operator to the output of the
previous one. In the case of translations, we can apply multiple convolutions
in sequence, since their outputs are all deﬁned on the same domain Ω. In the
general setting, since x ⋆θ is a function on G rather than on Ω, we cannot use
exactly the same operation subsequently—it means that the next operation


--- Page 47 ---
4. GEOMETRIC DOMAINS: THE 5 GS
43
has to deal with signals on G, i.e. x ∈X(G). Our deﬁnition of group convo-
lution allows this case: we take as domain Ω= G acted on by G itself via the
group action (g, h) 7→gh deﬁned by the composition operation of G. This
yields the representation ρ(g) acting on x ∈X(G) by (ρ(g)x)(h) = x(g−1h)
The representation of G
acting on functions deﬁned
on G itself is called the
regular representation of G.
.
Just like before, the inner product is deﬁned by integrating the point-wise
product of the signal and the ﬁlter over the domain, which now equals Ω= G.
In our example of spherical convolution, a second layer of convolution would
thus have the form
((x ⋆θ) ⋆φ)(R) =
Z
SO(3)
(x ⋆θ)(Q)φ(R−1Q)dQ.
Since convolution involves inner product that in turn requires integrating
over the domain Ω, we can only use it on domains Ωthat are small (in the
discrete case) or low-dimensional (in the continuous case). For instance,
we can use convolutions on the plane R2 (two dimensional) or special or-
thogonal group SE(3) (three dimensional), or on the ﬁnite set of nodes of
a graph (n-dimensional), but we cannot in practice perform convolution
on the group of permutations Σn, which has n! elements. Likewise, inte-
grating over higher-dimensional groups like the aﬃne group (containing
translations, rotations, shearing and scaling, for a total of 6 dimensions) is
not feasible in practice. Nevertheless, as we have seen in Section 5.3, we
can still build equivariant convolutions for large groups G by working with
signals deﬁned on low-dimensional spaces Ωon which G acts. Indeed, it is
possible to show that any equivariant linear map f : X(Ω) →X(Ω′) between
two domains Ω, Ω′ can be written as a generalised convolution similar to the
group convolution discussed here.
Second, we note that the Fourier transform we derived in the previous section
from the shift-equivariance property of the convolution can also be extended
to a more general case by projecting the signal onto the matrix elements of
irreducible representations of the symmetry group. We will discuss this in
future work. In the case of SO(3) studied here, this gives rise to the spherical
harmonics and Wigner D-functions, which ﬁnd wide applications in quantum
mechanics and chemistry.
Finally, we point to the assumption that has so far underpinned our discus-
sion in this section: whether Ωwas a grid, plane, or the sphere, we could
transform every point into any other point, intuitively meaning that all the
points on the domain “look the same.” A domain Ωwith such property is


--- Page 48 ---
44
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
called a homogeneous space, where for any u, v ∈Ωthere exists g ∈G such
that g.u = v
The additional properties,
e.u = u and g(h.u) = (gh).u
are tacitly assumed here.
. In the next section we will try to relax this assumption.
4.4
Geodesics and Manifolds
In our last example, the sphere S2
As well as the group of
rotations SO(3), by virtue of
it being a Lie group.
was a manifold, albeit a special one with a
global symmetry group due to its homogeneous structure. Unfortunately,
this is not the case for the majority of manifolds, which typically do not
have global symmetries. In this case, we cannot straightforwardly deﬁne an
action of G on the space of signals on Ωand use it to ‘slide’ ﬁlters around
in order to deﬁne a convolution as a direct generalisation of the classical
construction. Nevertheless, manifolds do have two types of invariance that
we will explore in this section: transformations preserving metric structure
and local reference frame change.
While for many machine learning readers manifolds might appear as some-
what exotic objects, they are in fact very common in various scientiﬁc do-
mains. In physics, manifolds play a central role as the model of our Universe
— according to Einstein’s General Relativity Theory, gravity arises from the
curvature of the space-time, modeled as a pseudo-Riemannian manifold. In
more ‘prosaic’ ﬁelds such as computer graphics and vision, manifolds are a
common mathematical model of 3D shapes.
The term ‘3D’ is somewhat
misleading and refers to the
embedding space. The shapes
themselves are 2D manifolds
(surfaces).
The broad spectrum of applica-
tions of such models ranges from virtual and augmented reality and special
eﬀects obtained by means of ‘motion capture’ to structural biology dealing
with protein interactions that stick together (‘bind’ in chemical jargon) like
pieces of 3D puzzle. The common denominator of these applications is the
use of a manifold to represent the boundary surface of some 3D object.
There are several reasons why such models are convenient.
The human body is an
example of a non-rigid object
deforming in a
nearly-isometric way.
First, they oﬀer a
compact description of the 3D object, eliminating the need to allocate memory
to ‘empty space’ as is required in grid-based representations. Second, they
allow to ignore the internal structure of the object. This is a handy property
for example in structural biology where the internal folding of a protein
molecule is often irrelevant for interactions that happen on the molecular
surface. Third and most importantly, one often needs to deal with deformable
objects that undergo non-rigid deformations. Our own body is one such
example, and many applications in computer graphics and vision, such as
the aforementioned motion capture and virtual avatars, require deformation
invariance. Such deformations can be modelled very well as transformations


--- Page 49 ---
4. GEOMETRIC DOMAINS: THE 5 GS
45
that preserve the intrinsic structure of a (Riemannian) manifold, namely the
distances between points measured along the manifold, without regard to
the way the manifold is embedded in the ambient space.
We should emphasise that manifolds fall under the setting of varying domains
in our Geometric Deep Learning blueprint, and in this sense are similar
to graphs. We will highlight the importance of the notion of invariance to
domain deformations – what we called ‘geometric stability’ in Section 3.3.
Since diﬀerential geometry is perhaps less familiar to the machine learning
audience, we will introduce the basic concepts required for our discussion
and refer the reader to Penrose (2005) for their detailed exposition.
Riemannian manifolds
Since the formal deﬁnition of a manifold
By ‘smooth’ we mean
diﬀerentiable suﬃent
number of times, which is
tacitly assumed for
convenience. ‘Deformed’
here means diﬀeomorphic, i.e.,
we can map between the two
neighbourhoods using a
smooth and invertible map
with smooth inverse.
is some-
what involved, we prefer to provide an intuitive picture at the expense of
some precision. In this context, we can think of a (diﬀerentiable or smooth)
manifold as a smooth multidimensional curved surface that is locally Eu-
clidean, in the sense that any small neighbourhood around any point it can
be deformed to a neighbourhood of Rs; in this case the manifold is said to
be s-dimensional. This allows us to locally approximate the manifold around
point u through the tangent space TuΩ. The latter can be visualised by think-
ing of a prototypical two-dimensional manifold, the sphere, and attaching a
plane to it at a point: with suﬃcient zoom, the spherical surface will seem
planar (Figure 11).
Formally, the tangent bundle
is the disjoint union
TΩ=
G
u∈Ω
TuΩ.
The collection of all tangent spaces is called the tangent
bundle, denoted TΩ; we will dwell on the concept of bundles in more detail
in Section 4.5.
A tangent vector, which we denote by X ∈TuΩ, can be thought of as a local
displacement from point u. In order to measure the lengths of tangent vectors
and angles between them,
A bilinear function g is said
to be positive-deﬁnite if
g(X, X) > 0 for any non-zero
vector X ̸= 0. If g is
expressed as a matrix G, it
means G ≻0. The
determinant |G|1/2 provides
a local volume element,
which does not depend on
the choice of the basis.
we need to equip the tangent space with additional
structure, expressed as a positive-deﬁnite bilinear function gu : TuΩ×TuΩ→
R depending smoothly on u. Such a function is called a Riemannian metric, in
honour of Bernhardt Riemann who introduced the concept in 1856, and can
be thought of as an inner product on the tangent space, ⟨X, Y ⟩u = gu(X, Y ),
which is an expression of the angle between any two tangent vectors X, Y ∈
TuΩ. The metric also induces a norm ∥X∥u = g1/2
u (X, X) allowing to locally
measure lengths of vectors.
We must stress that tangent vectors are abstract geometric entities that exists
in their own right and are coordinate-free. If we are to express a tangent vector


--- Page 50 ---
46
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
X numerically as an array of numbers, we can only represent it as a list of
coordinates x = (x1, . . . , xs) relative to some local basis
Unfortunately, too often
vectors are identiﬁed with
their coordinates. To
emphasise this important
diﬀerence, we use X to
denote a tangent vector and x
to denote its coordinates.
{X1, . . . Xs} ⊆TuΩ.
Similarly, the metric can be expressed as an s × s matrix G with elements
gij = gu(Xi, Xj) in that basis. We will return to this point in Section 4.5.
Figure 11: Basic notions of Riemannian geometry illustrated on the example
of the two-dimensional sphere S2 = {u ∈R3 : ∥u∥= 1}, realised a subset
(sub-manifold) of R3. The tangent space to the sphere is given as TuS2 =
{x ∈R3 : x⊤u = 0} and is a 2D plane – hence this is a 2-dimensional
manifold. The Riemannian metric is simply the Euclidean inner product
restricted to the tangent plane, ⟨x, y⟩u = x⊤y for any x, x ∈TuS2. The
exponential map is given by expu(x) = cos(∥x∥)u + sin(∥x∥)
∥x∥
x, for x ∈TuS2.
Geodesics are great arcs of length d(u, v) = cos−1(u⊤v).
A manifold equipped with a metric is called a Riemannian manifold and
properties that can be expressed entirely in terms of the metric are said
to be intrinsic. This is a crucial notion for our discussion, as according to
our template, we will be seeking to construct functions acting on signals
deﬁned on Ωthat are invariant to metric-preserving transformations called
isometries that deform the manifold without aﬀecting its local structure.
This result is known as the
Embedding Theorem, due to
Nash (1956). The art of
origami is a manifestation of
diﬀerent isometric
embeddings of the planar
surface in R3 (Figure:
Shutterstock/300 librarians).
If
such functions can be expressed in terms of intrinsic quantities, they are
automatically guaranteed to be isometry-invariant and thus unaﬀected by
isometric deformations. These results can be further extended to dealing
with approximate isometries; this is thus an instance of the geometric stability
(domain deformation) discussed in our blueprint.
While, as we noted, the deﬁnition of a Riemannian manifold does not require
a geometric realisation in any space, it turns out that any smooth Riemannian
manifold can be realised as a subset of a Euclidean space of suﬃciently high
dimension (in which case it is said to be ‘embedded’ in that space) by using


--- Page 51 ---
4. GEOMETRIC DOMAINS: THE 5 GS
47
the structure of the Euclidean space to induce a Riemannian metric. Such an
embedding is however not necessarily unique – as we will see, two diﬀerent
isometric realisations of a Riemannian metric are possible.
Scalar and Vector ﬁelds
Since we are interested in signals deﬁned on Ω,
we need to provide the proper notion of scalar- and vector-valued functions
on manifolds. A (smooth) scalar ﬁeld is a function of the form x : Ω→R.
Example of a scalar ﬁeld.
Scalar ﬁelds form a vector space X(Ω, R) that can be equipped with the inner
product
⟨x, y⟩=
Z
Ω
x(u)y(u)du,
(15)
where du is the volume element induced by the Riemannian metric. A
(smooth) tangent vector ﬁeld is a function of the form X : Ω→TΩassigning to
each point a tangent vector in the respective tangent space, u 7→X(u) ∈TuΩ.
Vector ﬁelds
Example of a vector ﬁeld.
The ﬁelds are typically
assumed to be of the same
regularity class (smoothness)
as the manifold itself.
also form a vector space X(Ω, TΩ) with the inner product
deﬁned through the Riemannian metric,
⟨X, Y ⟩=
Z
Ω
gu(X(u), Y (u))du.
(16)
Intrinsic gradient
Another way to think of (and actually deﬁne) vector
ﬁelds is as a generalised notion of derivative. In classical calculus, one
can locally linearise a (smooth) function through the diﬀerential dx(u) =
x(u + du) −x(u), which provides the change of the value of the function
x at point u as a result of an iniﬁnitesimal displacement du. However, in
our case the naïve use of this deﬁnition is impossible, since expressions of
the form “u + du” are meaningless on manifolds due to the lack of a global
vector space structure.
The solution is to use tangent vectors as a model of local inﬁnitesimal displace-
ment. Given a smooth scalar ﬁeld x ∈X(Ω, R), we can think of a (smooth)
vector ﬁeld as a linear map Y : X(Ω, R) →X(Ω, R) satisfying the properties
of a derivation: Y (c) = 0 for any constant c (corresponding to the intuition
that constant functions have vanishing derivatives), Y (x + z) = Y (x) + Y (z)
(linearity), and Y (xz) = Y (x)z + xY (z) (product or Leibniz rule), for any
smooth scalar ﬁelds x, z ∈X(Ω, R). It can be shown that one can use these
properties to deﬁne vector ﬁelds axiomatically. The diﬀerential dx(Y ) = Y (x)
can be viewed as an operator (u, Y ) 7→Y (x) and interpreted as follows: the


--- Page 52 ---
48
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
change of x as the result of displacement Y ∈TuΩat point u is given by
dux(Y ).
Importantly, this construction
does not use the Riemannian
metric whatsoever and can
thus can be extended to a
more general construction of
bundles discussed in the
Section 4.5.
It is thus an extension of the classical notion of directional derivative.
Alternatively, at each point u the diﬀerential can be regarded as a linear
functional dxu : TuΩ→R acting on tangent vectors X ∈TuΩ. Linear
functionals on a vector space are called dual vectors or covectors; if in addition
we are given an inner product (Riemannian metric), a dual vector can always
be represented as
dxu(X) = gu(∇x(u), X).
This is a consequence of the
Riesz-Fréchet Representation
Theorem, by which every
dual vector can be expressed
as an inner product with a
vector.
The representation of the diﬀerential at point u is a tangent vector ∇x(u) ∈
TuΩcalled the (intrinsic) gradient of x; similarly to the gradient in classical
calculus, it can be thought of as the direction of the steepest increase of x.
The gradient considered as an operator ∇: X(Ω, R) →X(Ω, TΩ) assigns at
each point x(u) 7→∇x(u) ∈TuΩ; thus, the gradient of a scalar ﬁeld x is a
vector ﬁeld ∇x.
Geodesics
Now consider a smooth curve γ : [0, T] →Ωon the manifold
with endpoints u = γ(0) and v = γ(T). The derivative of the curve at point
t is a tangent vector γ′(t) ∈Tγ(t)Ωcalled the velocity vector.
It is tacitly assumed that
curves are given in arclength
parametrisation, such that
∥γ′∥= 1 (constant velocity).
Among all the
curves connecting points u and v, we are interested in those of minimum
length, i.e., we are seeking γ minimising the length functional
ℓ(γ) =
Z T
0
∥γ′(t)∥γ(t)dt =
Z T
0
g1/2
γ(t)(γ′(t), γ′(t))dt.
Such curves are called geodesics (from the Greek γεοδαισία, literally ‘division
of Earth’) and they play important role in diﬀerential geometry. Crucially to
our discussion, the way we deﬁned geodesics is intrinsic, as they depend
solely on the Riemannian metric (through the length functional).
Readers familiar with diﬀerential geometry might recall that geodesics are a
more general concept and their deﬁnition in fact does not necessarily require
a Riemannian metric but a connection (also called a covariant derivative, as
it generalises the notion of derivative to vector and tensor ﬁelds), which is
deﬁned axiomatically, similarly to our construction of the diﬀerential. Given
a Riemannian metric, there exists a unique special connection called the
The Levi-Civita connection is
torsion-free and compatible
with the metric. The
Fundamental Theorem of
Riemannian geometry
guarantees its existence and
uniqueness.
Levi-
Civita connection which is often tacitly assumed in Riemannian geometry.
Geodesics arising from this connection are the length-minimising curves we
have deﬁned above.


--- Page 53 ---
4. GEOMETRIC DOMAINS: THE 5 GS
49
We will show next how to use geodesics to deﬁne a way to transport tangent
vectors on the manifold (parallel transport), create local intrinsic maps from
the manifold to the tangent space (exponential map), and deﬁne distances
(geodesic metric). This will allow us to construct convolution-like operations
by applying a ﬁlter locally in the tangent space.
Euclidean transport of a
vector from A to C makes no
sense on the sphere, as the
resulting vectors (red) are
not in the tangent plane.
Parallel transport from A to C
(blue) rotates the vector
along the path. It is path
dependent: going along the
path BC and ABC produces
diﬀerent results.
Parallel transport
One issue we have already encountered when dealing
with manifolds is that we cannot directly add or subtract two points u, v ∈Ω.
The same problem arises when trying to compare tangent vectors at diﬀerent
points: though they have the same dimension, they belong to diﬀerent spaces,
e.g. X ∈TuΩand Y ∈TvΩ, and thus not directly comparable. Geodesics
provide a mechanism to move vectors from one point to another, in the
following way: let γ be a geodesic connecting points u = γ(0) and v = γ(T)
and let X ∈TuΩ. We can deﬁne a new set of tangent vectors along the
geodesic, X(t) ∈Tγ(t)Ωsuch that the length of X(t) and the angle (expressed
through the Riemannian metric) between it and the velocity vector of the
curve is constant,
gγ(t)(X(t), γ′(t)) = gu(X, γ′(0)) = const,
∥X(t)∥γ(t) = ∥X∥u = const.
As a result, we get a unique vector X(T) ∈TvΩat the end point v.
The map Γu→v(X) : TuΩ→TuΩand TvΩdeﬁned as Γu→v(X) = X(T)
using the above notation is called parallel transport or connection; the latter
term implying it is a mechanism to ‘connect’ between the tangent spaces
TuΩand TvΩ. Due to the angle and length preservation conditions, parallel
transport amounts to only rotation of the vector, so it can be associated with
an element of the special orthogonal group SO(s) (called the structure group
of the tangent bundle),
Assuming that the manifold
is orientable, otherwise O(s).
which we will denote by gu→v and discuss in further
detail in Section 4.5.
As we mentioned before, a connection can be deﬁned axiomatically inde-
pendently of the Riemannian metric, providing thus an abstract notion of
parallel transport along any smooth curve. The result of such transport,
however, depends on the path taken.
Exponential map
Locally around a point u, it is always possible to deﬁne
a unique geodesic in a given direction X ∈TuΩ, i.e. such that γ(0) = u and
γ′(0) = X. When γX(t) is deﬁned for all t ≥0 (that is, we can shoot the


--- Page 54 ---
50
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
geodesic from a point u for as long as we like), the manifold is said to be
geodesically complete and the exponential map is deﬁned on the whole tangent
space. Since compact manifolds are geodesically complete, we can tacitly
assume this convenient property.
This deﬁnition of geodesic provided a point and a direction gives a natural
mapping from (a subset of) the tangent space TuΩto Ωcalled the exponential
map
Note that geodesic
completeness does not
necessarily guarantee that
exp is a global
diﬀeomorphism – the largest
radius r about u for which
expu(Br(0) ⊆TuΩ) is
mapped diﬀeomorphically is
called the injectivity radius.
exp : Br(0) ⊂TuΩ→Ω, which is deﬁned by taking a unit step along
the geodesic in the direction X, i.e., expu(X) = γX(1). The exponential map
expu is a local diﬀeomorphism, as it deforms the neighbourhood Br(0) (a
ball or radius r) of the origin on TuΩinto a neighbourhood of u. Conversely,
one can also regard the exponential map as an intrinsic local deformation
(‘ﬂattening’) of the manifold into the tangent space.
Geodesic distances
A result known as the Hopf-Rinow Theorem
Hopf-Rinow Theorem thus
estabilishes the equivalence
between geodesic and metric
completeness, the latter
meaning every Cauchy
sequence converges in the
geodesic distance metric.
guar-
antees that geodesically complete manifolds are also complete metric spaces,
in which one can realise a distance (called the geodesic distance or metric)
between any pair of points u, v as the length of the shortest path between
them
dg(u, v) = min
γ
ℓ(γ)
s.t.
γ(0) = u, γ(T) = v,
which exists (i.e., the minimum is attained).
Note that the term ‘metric’ is
used in two senses:
Riemannian metric g and
distance d. To avoid
confusion, we will use the
term ‘distance’ referring to
the latter. Our notation dg
makes the distance depend
on the Riemannian metric g,
though the deﬁnition of
geodesic length L.
Isometries
Consider now a deformation of our manifold Ωinto another
manifold ˜Ωwith a Riemannian metric h, which we assume to be a dif-
feomorphism η : (Ω, g) →(˜Ω, h) between the manifolds. Its diﬀerential
dη : TΩ→T ˜Ωdeﬁnes a map between the respective tangent bundles (re-
ferred to as pushforward), such that at a point u, we have dηu : TuΩ→Tη(u) ˜Ω,
interpreted as before: if we make a small displacement from point u by
tangent vector X ∈TuΩ, the map η will be displaced from point η(u) by
tangent vector dηu(X) ∈Tη(u) ˜Ω.
Since the pushforward
Pushforward and pullback
are adjoint operators
⟨η∗α, X⟩= ⟨α, η∗X⟩where
α ∈T ∗Ωis a dual vector ﬁeld,
deﬁned at each point as a
linear functional acting on
TuΩand the inner products
are deﬁned respectively on
vector and dual vector ﬁelds.
provides a mechanism to associate tangent vectors
on the two manifolds, it allows to pullback the metric h from ˜Ωto Ω,
(η∗h)u(X, Y ) = hη(u)(dηu(X), dηu(Y ))
If the pullback metric coincides at every point with that of Ω, i.e., g = η∗h,
the map η is called (a Riemannian) isometry. For two-dimensional manifolds


--- Page 55 ---
4. GEOMETRIC DOMAINS: THE 5 GS
51
(surfaces), isometries can be intuitively understood as inelastic deformations
that deform the manifold without ‘stretching’ or ‘tearing’ it.
By virtue of their deﬁnition, isometries preserve intrinsic structures such as
geodesic distances, which are expressed entirely in terms of the Riemannian
metric. Therefore, we can also understand isometries from the position of
metric geometry, as distance-preserving maps (‘metric isometries’) between
metric spaces η : (Ω, dg) →(˜Ω, dh), in the sense that
dg(u, v) = dh(η(u), η(v))
for all u, v ∈Ω, or more compactly, dg = dh ◦(η × η). In other words,
Riemannian isometries are also metric isometries. On connected manifolds,
the converse is also true: every metric isometry is also a Riemannian isometry.
This result is known as the
Myers–Steenrod Theorem.
We tacitly assume our
manifolds to be connected.
In our Geometric Deep Learning blueprint, η is a model of domain defor-
mations. When η is an isometry, any intrinsic quantities are unaﬀected by
such deformations. One can generalise exact (metric) isometries through
the notions of metric dilation
dil(η) = sup
u̸=v∈Ω
dh(η(u), η(v))
dg(u, v)
or metric distortion
dis(η) = sup
u,v∈Ω
|dh(η(u), η(v)) −dg(u, v)|,
The Gromov-Hausdorﬀ
distance between metric
spaces, which we mentioned
in Section 3.2, can be
expressed as the smallest
possible metric distortion.
which capture the relative and absolute change of the geodesic distances
under η, respectively. The condition (5) for the stability of a function f ∈
F(X(Ω)) under domain deformation can be rewritten in this case as
∥f(x, Ω) −f(x ◦η−1, ˜Ω)∥≤C∥x∥dis(η).
Intrinsic symmetries
A particular case of the above is a diﬀeomorphism
of the domain itself (what we termed automorphism in Section 3.2), which we
will denote by τ ∈Diﬀ(Ω). We will call it a Riemannian (self-)isometry if the
pullback metric satisﬁes τ ∗g = g, or a metric (self-)isometry if dg = dg◦(τ×τ).
Not surprisingly,
Continuous symmetries on
manifolds are inﬁnitesimally
generated by special tangent
vector ﬁelds called Killing
ﬁelds, named after Wilhelm
Killing.
isometries form a group with the composition operator
denoted by Iso(Ω) and called the isometry group; the identity element is
the map τ(u) = u and the inverse always exists (by deﬁnition of τ as a
diﬀeomorphism). Self-isometries are thus intrinsic symmetries of manifolds.


--- Page 56 ---
52
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
Fourier analysis on Manifolds
We will now show how to construct intrin-
sic convolution-like operations on manifolds, which, by construction, will be
invariant to isometric deformations. For this purpose, we have two options:
One is to use an analogy of the Fourier transform, and deﬁne the convolution
as a product in the Fourier domain. The other is to deﬁne the convolution
spatially, by correlating a ﬁlter locally with the signal. Let us discuss the
spectral approach ﬁrst.
We remind that in the Euclidean domain the Fourier transform is obtained as
the eigenvectors of circulant matrices, which are jointly diagonalisable due to
their commutativity. Thus, any circulant matrix and in particular, diﬀerential
operator, can be used to deﬁne an analogy of the Fourier transform on general
domains. In Riemannian geometry, it is common to use the orthogonal
eigenbasis of the Laplacian operator, which we will deﬁne here.
For this purpose, recall our deﬁnition of the intrinsic gradient operator
∇: X(Ω, R) →X(Ω, TΩ), producing a tangent vector ﬁeld that indicates
the local direction of steepest increase of a scalar ﬁeld on the manifold. In
a similar manner, we can deﬁne the divergence operator ∇∗: X(Ω, TΩ) →
X(Ω, R). If we think of a tangent vector ﬁeld as a ﬂow on the manifold, the
divergence measures the net ﬂow of a ﬁeld at a point, allowing to distinguish
between ﬁeld ‘sources’ and ‘sinks’. We use the notation ∇∗(as opposed to
the common div) to emphasise that the two operators are adjoint,
⟨X, ∇x⟩= ⟨∇∗X, x⟩,
where we use the inner products (15) and (16) between scalar and vector
ﬁelds.
The Laplacian (also known as the Laplace-Beltrami operator in diﬀerential geom-
etry) is an operator on X(Ω) deﬁned as ∆= ∇∗∇, which can be interpreted
From this interpretation it is
also clear that the Laplacian
is isotropic. We will see in
Section 4.6 that it is possible
to deﬁne anisotropic Laplacians
(see (Andreux et al., 2014;
Boscaini et al., 2016b)) of the
form ∇∗(A(u)∇), where
A(u) is a position-dependent
tensor determining local
direction.
as the diﬀerence between the average of a function on an inﬁnitesimal sphere
around a point and the value of the function at the point itself. It is one
of the most important operators in mathematical physics, used to describe
phenomena as diverse as heat diﬀusion, quantum oscillations, and wave
propagation. Importantly in our context, the Laplacian is intrinsic, and thus
invariant under isometries of Ω.
It is easy to see that the Laplacian is self-adjoint (‘symmetric’),
⟨∇x, ∇x⟩= ⟨x, ∆x⟩= ⟨∆x, x⟩.


--- Page 57 ---
4. GEOMETRIC DOMAINS: THE 5 GS
53
The quadratic form on the left in the above expression is actually the already
familiar Dirichlet energy,
c2(x) = ∥∇x∥2 = ⟨∇x, ∇x⟩=
Z
Ω
∥∇x(u)∥2
udu =
Z
Ω
gu(∇x(u), ∇x(u))du
measuring the smoothness of x.
The Laplacian operator admits an eigedecomposition
∆ϕk = λkϕk,
k = 0, 1, . . .
with countable spectrum if the manifold is compact (which we tacitly as-
sume), and orthogonal eigenfunctions, ⟨ϕk, ϕl⟩= δkl, due to the self-adjointness
of ∆. The Laplacian eigenbasis can also be constructed as a set of orthogonal
minimisers of the Dirichlet energy,
ϕk+1 = arg min
ϕ ∥∇ϕ∥2
s.t.
∥ϕ∥= 1 and ⟨ϕ, ϕj⟩= 0
for j = 0, . . . , k, allowing to interpret it as the smoothest orthogonal basis
on Ω. The eigenfunctions ϕ0, ϕ1, . . . and the corresponding eigenvalues
0 = λ0 ≤λ1 ≤. . . can be interpreted as the analogy of the atoms and
frequencies in the classical Fourier transform.
In fact eiξu are the
eigenfunctions of the
Euclidean Laplacian
d2
du2 .
This orthogonal basis allows to expand square-integrable functions on Ω
into Fourier series
x(u) =
X
k≥0
⟨x, ϕk⟩ϕk(u)
where ˆxk = ⟨x, ϕk⟩are referred to as the Fourier coeﬃcient or the (gener-
alised) Fourier transform of x.
Note that this Fourier
transform has a discrete
index, since the spectrum is
discrete due to the
compactness of Ω.
Truncating the Fourier series results in an
approximation error that can be bounded (Aﬂalo and Kimmel, 2013) by
x −
N
X
k=0
⟨x, ϕk⟩ϕk

2
≤∥∇x∥2
λN+1
.
Aﬂalo et al. (2015) further showed that no other basis attains a better error,
making the Laplacian eigenbasis optimal for representing smooth signals on
manifolds.


--- Page 58 ---
54
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
Spectral Convolution on Manifolds
Spectral convolution can be deﬁned as
the product of Fourier transforms of the signal x and the ﬁlter θ,
(x ⋆θ)(u) =
X
k≥0
(ˆxk · ˆθk)ϕk(u).
(17)
Note that here we use what is a property of the classical Fourier transform
(the Convolution Theorem) as a way to deﬁne a non-Euclidean convolution.
By virtue of its construction, the spectral convolution is intrinsic and thus
isometry-invariant. Furthermore, since the Laplacian operator is isotropic,
it has no sense of direction; in this sense, the situation is similar to that we
had on graphs in Section 4.1 due to permutation invariance of neighbour
aggregation.
Figure 12: Instability of spectral ﬁlters under domain perturbation. Left: a
signal x on the mesh Ω. Middle: result of spectral ﬁltering in the eigenbasis
of the Laplacian ∆on Ω. Right: the same spectral ﬁlter applied to the
eigenvectors of the Laplacian ˜∆of a nearly-isometrically perturbed domain
˜Ωproduces a very diﬀerent result.
In practice, a direct computation of (17) appears to be prohibitively expensive
due to the need to diagonalise the Laplacian. Even worse, it turns out
geometrically unstable: the higher-frequency eigenfunctions of the Laplacian
can change dramatically as a result of even small near-isometric perturbations
of the domain Ω(see Figure 12). A more stable solution is provided by
realising the ﬁlter as a spectral transfer function of the form ˆp(∆),
(ˆp(∆)x)(u)
=
X
k≥0
ˆp(λk)⟨x, ϕk⟩ϕk(u)
(18)
=
Z
Ω
x(v)
X
k≥0
ˆp(λk)ϕk(v)ϕk(u) dv
(19)
which can be interpreted in two manners: either as a spectral ﬁlter (18),
where we identify ˆθk = ˆp(λk), or as a spatial ﬁlter (19) with a position-
dependent kernel θ(u, v) = P
k≥0 ˆp(λk)ϕk(v)ϕk(u). The advantage of this


--- Page 59 ---
4. GEOMETRIC DOMAINS: THE 5 GS
55
formulation is that ˆp(λ) can be parametrised by a small number of coeﬃcients,
and choosing parametric functions such as polynomials
Geometric Deep Learning
methods based on spectral
convolution expressed
through the Fourier
transform are often referred
to as ‘spectral’ and opposed
to ‘spatial’ methods we have
seen before in the context of
graphs. We see here that
these two views may be
equivalent, so this dichotomy
is somewhat artiﬁcial and not
completely appropriate.
ˆp(λ) = Pr
l=0 αlλl
allows for eﬃciently computing the ﬁlter as
(ˆp(∆)x)(u) =
X
k≥0
r
X
l=0
αlλl
k ⟨x, ϕk⟩ϕk(u) =
r
X
l=0
αl(∆lx)(u),
avoiding the spectral decomposition altogether. We will discuss this con-
struction in further detail in Section 4.6.
Spatial Convolution on Manifolds
A second alternative is to attempt
deﬁning convolution on manifolds is by matching a ﬁlter at diﬀerent points,
like we did in formula (14),
(x ⋆θ)(u) =
Z
TuΩ
x(expu Y )θu(Y )dY,
(20)
where we now have to use the exponential map to access the values of the
scalar ﬁeld x from the tangent space, and the ﬁlter θu is deﬁned in the tangent
space at each point and hence position-dependent. If one deﬁnes the ﬁlter
intrinsically, such a convolution would be isometry-invariant, a property we
mentioned as crucial in many computer vision and graphics applications.
We need, however, to note several substantial diﬀerences from our previous
construction in Sections 4.2–4.3. First, because a manifold is generally not
a homogeneous space, we do not have anymore a global group structure
allowing us have a shared ﬁlter (i.e., the same θ at every u rather than θu
in expression (20)) deﬁned at one point and then move it around. An
analogy of this operation on the manifold would require parallel transport,
allowing to apply a shared θ, deﬁned as a function on TuΩ, at some other
TvΩ. However, as we have seen, this in general will depend on the path
between u and v, so the way we move the ﬁlter around matters.Third, since
we can use the exponential map only locally, the ﬁlter must be local, with
support bounded by the injectivity radius. Fourth and most crucially, we
cannot work with θ(X), as X is an abstract geometric object: in order for it
to be used for computations, we must represent it relative to some local basis
ωu : Rs →TuΩ, as an s-dimensional array of coordinates x = ω−1
u (X). This
allows us to rewrite the convolution (20) as
(x ⋆θ)(u) =
Z
[0,1]s x(expu(ωuy))θ(y)dy,
(21)


--- Page 60 ---
56
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
with the ﬁlter deﬁned on the unit cube. Since the exponential map is intrinsic
(through the deﬁnition of geodesic), the resulting convolution is isometry-
invariant.
Yet, this tacitly assumed we can carry the frame ωu along to another manifold,
i.e. ω′
u = dηu ◦ωu. Obtaining such a frame (or gauge, in physics terminology)
given only a the manifold Ωin a consistent manner is however fraught with
diﬃculty. First, a smooth global gauge may not exist: this is the situation on
manifolds that are not parallelisable,
The sphere S2 is an example
of a non-parallelisable
manifold, a result of the
Poincaré-Hopf Theorem, which
is colloquially stated as ‘one
cannot comb a hairy ball
without creating a cowlick.’
in which case one cannot deﬁne a smooth
non-vanishing tangent vector ﬁeld. Second, we do not have a canonical gauge
on manifolds, so this choice is arbitrary; since our convolution depends on
ω, if one chose a diﬀerent one, we would obtain diﬀerent results.
We should note that this is a case where practice diverges from theory: in
practice, it is possible to build frames that are mostly smooth, with a limited
number of singularities, e.g. by taking the intrinsic gradient of some intrinsic
scalar ﬁeld on the manifold.
Example of stable gauges
constructed on
nearly-isometric manifolds
(only one axis is shown)
using the GFrames algorithm
of Melzi et al. (2019).
Moreover, such constructions are stable, i.e.,
the frames constructed this way will be identical on isometric manifolds
and similar on approximately isometric ones. Such approaches were in fact
employed in the early works on deep learning on manifolds (Masci et al.,
2015; Monti et al., 2017).
Nevertheless, this solution is not entirely satisfactory because near singu-
larities, the ﬁlter orientation (being deﬁned in a ﬁxed manner relative to
the gauge) will vary wildly, leading to a non-smooth feature map even if
the input signal and ﬁlter are smooth. Moreover, there is no clear reason
why a given direction at some point u should be considered equivalent to
another direction at an altogether diﬀerent point v. Thus, despite practical
alternatives, we will look next for a more theoretically well-founded approach
that would be altogether independent on the choice of gauge.
4.5
Gauges and Bundles
The notion of gauge, which we have deﬁned as a frame for the tangent space,
is quite a bit more general in physics: it can refer to a frame for any
Historically, ﬁbre bundles
arose ﬁrst in modern
diﬀerential geometry of Élie
Cartan (who however did not
deﬁne them explicitly), and
were then further developed
as a standalone object in the
ﬁeld of topology in the 1930s.
vector
bundle, not just the tangent bundle. Informally, a vector bundle describes
a family of vector spaces parametrised by another space and consists of a
base space Ωwith an identical vector space V (called the ﬁbre) attached to
each position u ∈Ω(for the tangent bundle these are the tangent spaces


--- Page 61 ---
4. GEOMETRIC DOMAINS: THE 5 GS
57
TuΩ). Roughly speaking, a bundle looks as a product Ω× V locally around
u, but globally might be ‘twisted’ and have an overall diﬀerent structure. In
Geometric Deep Learning, ﬁbres can be used to model the feature spaces at
each point in the manifold Ω, with the dimension of the ﬁbre being equal to
the number of feature channels. In this context, a new and fascinating kind
of symmetry, called gauge symmetry may present itself.
Let us consider again an s-dimensional manifold Ωwith its tangent bundle
TΩ, and a vector ﬁeld X : Ω→TΩ(which in this terminology is referred
to as a section on the tangent bundle). Relative to a gauge ω for the tangent
bundle, X is represented as a function x : Ω→Rs. However it is important
to realise that what we are really interested in is the underlying geometrical
object (vector ﬁeld), whose representation as a function x ∈X(Ω, Rs) depends
on the choice of gauge ω. If we change the gauge, we also need to change x so
as to preserve the underlying vector ﬁeld being represented.
Tangent bundles and the Structure group
When we change the gauge, we
need to apply at each point an invertible matrix that maps the old gauge to
the new one. This matrix is unique for every pair of gauges at each point, but
possibly diﬀerent at diﬀerent points. In other words, a gauge transformation is
a mapping g : Ω→GL(s), where GL(s) is the general linear group of invertible
s × s matrices. It acts on the gauge ωu : Rs →TuΩto produce a new gauge
ω′
u = ωu ◦gu : Rs →TuΩ. The gauge transformation acts on a coordinate
vector ﬁeld at each point via x′(u) = g−1
u x(u) to produce the coordinate
representation x′ of X relative to the new gauge. The underlying vector ﬁeld
remains unchanged:
X(u) = ω′
u(x′(u)) = ωu(gug−1
u x(u)) = ωu(x(u)) = X(u),
which is exactly the property we desired. More generally, we may have a
ﬁeld of geometric quantities that transform according to a representation
ρ of GL(s), e.g. a ﬁeld of 2-tensors (matrices) A(u) ∈Rs×s that transform
like A′(u) = ρ2(g−1
u )A(u) = ρ1(gu)A(u)ρ1(g−1
u ). In this case, the gauge
transformation gu acts via ρ(gu).
Sometimes we may wish to restrict attention to frames with a certain property,
such as orthogonal frames, right-handed frames, etc. Unsurprisingly, we are
interested in a set of some property-preserving transformations that form a
group. For instance, the group that preserves orthogonality is the orthogonal
group O(s) (rotations and reﬂections), and the group that additionally


--- Page 62 ---
58
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
preserves orientation or ‘handedness’ is SO(s) (pure rotations). Thus, in
general we have a group G called the structure group of the bundle, and a
gauge transformation is a map g : Ω→G. A key observation is that in all
cases with the given property, for any two frames at a given point there exists
exactly one gauge transformation relating them.
As mentioned before, gauge theory extends beyond tangent bundles, and
in general, we can consider a bundle of vector spaces whose structure and
dimensions are not necessarily related to those of the base space Ω.
We use s to denote the
dimension of the base space
Ωand d referring to the
dimension of the ﬁbre. For
tangent bundles, d = s is the
dimension of the underlying
manifold. For RGB image,
s = 2 and d = 3.
For
instance, a color image pixel has a position u ∈Ω= Z2 on a 2D grid and
a value x(u) ∈R3 in the RGB space, so the space of pixels can be viewed
as a vector bundle with base space Z2 and a ﬁbre R3 attached at each point.
It is customary to express an RGB image relative to a gauge that has basis
vectors for R, G, and B (in that order), so that the coordinate representation
of the image looks like x(u) = (r(u), g(u), b(u))⊤. But we may equally well
permute the basis vectors (color channels) independently at each position,
as long as we remember the frame (order of channels) in use at each point
In this example we have
chosen G = Σ3 the
permutations of the 3 color
channels as the structure
group of the bundle. Other
choices, such as a Hue
rotation G = SO(2) are also
possible.
.
As a computational operation this is rather pointless, but as we will see
shortly it is conceptually useful to think about gauge transformations for the
space of RGB colors, because it allows us to express a gauge symmetry – in
this case, an equivalence between colors – and make functions deﬁned on
images respect this symmetry (treating each color equivalently).
As in the case of a vector ﬁeld on a manifold, an RGB gauge transforma-
tion changes the numerical representation of an image (permuting the RGB
values independently at each pixel) but not the underlying image. In ma-
chine learning applications, we are interested in constructing functions
f ∈F(X(Ω)) on such images (e.g. to perform image classiﬁcation or seg-
mentation), implemented as layers of a neural network. It follows that if, for
whatever reason, we were to apply a gauge transformation to our image, we
would need to also change the function f (network layers) so as to preserve
their meaning. Consider for simplicity a 1 × 1 convolution, i.e. a map that
takes an RGB pixel x(u) ∈R3 to a feature vector y(u) ∈RC. According
to our Geometric Deep Learning blueprint, the output is associated with
a group representation ρout, in this case a C-dimensional representation of
the structure group G = Σ3 (RGB channel permutations), and similarly the
input is associated with a representation ρin(g) = g. Then, if we apply a
gauge transformation to the input, we would need to change the linear map
(1×1 convolution) f : R3 →RC to f′ = ρ−1
out(g)◦f ◦ρin(g) so that the output
feature vector y(u) = f(x(u)) transforms like y′(u) = ρout(gu)y(u) at every


--- Page 63 ---
4. GEOMETRIC DOMAINS: THE 5 GS
59
point. Indeed we verify:
y′ = f′(x′) = ρ−1
out(g)f(ρin(g)ρ−1
in (g)x) = ρ−1
out(g)f(x).
Here the notation ρ−1(g)
should be understood as the
inverse of the group
representation (matrix) ρ(g).
Gauge Symmetries
To say that we consider gauge transformations to be
symmetries is to say that any two gauges related by a gauge transformation
are to be considered equivalent. For instance, if we take G = SO(d), any
two right-handed orthogonal frames are considered equivalent, because we
can map any such frame to any other such frame by a rotation. In other
words, there are no distinguished local directions such as “up” or “right”.
Similarly, if G = O(d) (the orthogonal group), then any left and right handed
orthogonal frame are considered equivalent. In this case, there is no preferred
orientation either. In general, we can consider a group G and a collection
of frames at every point u such that for any two of them there is a unique
g(u) ∈G that maps one frame onto the other.
Regarding gauge transformations as symmetries in our Geometric Deep
Learning blueprint, we are interested in making the functions f acting on
signals deﬁned on Ωand expressed with respect to the gauge should equiv-
ariant to such transformation. Concretely, this means that if we apply a gauge
transformation to the input, the output should undergo the same transforma-
tion (perhaps acting via a diﬀerent representation of G). We noted before that
when we change the gauge, the function f should be changed as well, but for
a gauge equivariant map this is not the case: changing the gauge leaves the
mapping invariant. To see this, consider again the RGB color space example.
The map f : R3 →RC is equivariant if f ◦ρin(g) = ρout(g)◦f, but in this case
the gauge transformation applied to f has no eﬀect: ρ−1
out(g) ◦f ◦ρin(g) = f.
In other words, the coordinate expression of a gauge equivariant map is
independent of the gauge, in the same way that in the case of graph, we
applied the same function regardless of how the input nodes were permuted.
However, unlike the case of graphs and other examples covered so far, gauge
transformations act not on Ωbut separately on each of the feature vectors x(u)
by a transformation g(u) ∈G for each u ∈Ω.
Further considerations enter the picture when we look at ﬁlters on manifolds
with a larger spatial support. Let us ﬁrst consider an easy example of a
mapping f : X(Ω, R) →X(Ω, R) from scalar ﬁelds to scalar ﬁelds on an
s-dimensional manifold Ω. Unlike vectors and other geometric quantities,
scalars do not have an orientation, so a scalar ﬁeld x ∈X(Ω, R) is invariant to
gauge transformations (it transforms according to the trivial representation


--- Page 64 ---
60
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
ρ(g) = 1). Hence, any linear map from scalar ﬁelds to scalar ﬁelds is gauge
equivariant (or invariant, which is the same in this case). For example, we
could write f similarly to (19), as a convolution-like operation with a position-
dependent ﬁlter θ : Ω× Ω→R,
(x ⋆θ)(u) =
Z
Ω
θ(u, v)x(v)dv.
(22)
This implies that we have a potentially diﬀerent ﬁlter θu = θ(u, ·) at each
point, i.e., no spatial weight sharing — which gauge symmetry alone does
not provide.
Consider now a more interesting case of a mapping f : X(Ω, TΩ) →X(Ω, TΩ)
from vector ﬁelds to vector ﬁelds. Relative to a gauge, the input and output
vector ﬁelds X, Y ∈X(Ω, TΩ) are vector-valued functions x, y ∈X(Ω, Rs).
A general linear map between such functions can be written using the same
equation we used for scalars (22), only replacing the scalar kernel by a
matrix-valued one Θ : Ω× Ω→Rs×s. The matrix Θ(u, v) should map tan-
gent vectors in TvΩto tangent vectors in TuΩ, but these points have diﬀerent
gauges that we may change arbitrarily and independently. That is, the ﬁlter
would have to satisfy Θ(u, v) = ρ−1(g(u))Θ(u, v)ρ(g(v)) for all u, v ∈Ω,
where ρ denotes the action of G on vectors, given by an s × s rotation matrix.
Since g(u) and g(v) can be chosen freely, this is an overly strong constraint
on the ﬁlter.
Indeed Θ would have to be
zero in this case
A better approach is to ﬁrst transport the vectors to a common tangent space
by means of the connection, and then impose gauge equivariance w.r.t. a
single gauge transformation at one point only. Instead of (22), we can then
deﬁne the following map between vector ﬁelds,
(x ⋆Θ)(u) =
Z
Ω
Θ(u, v)ρ(gv→u)x(v)dv,
(23)
where gv→u ∈G denotes the parallel transport from v to u along the geodesic
connecting these two points; its representation ρ(gv→u) is an s × s rotation
matrix rotating the vector as it moves between the points. Note that this
geodesic is assumed to be unique, which is true only locally and thus the
ﬁlter must have a local support. Under a gauge transformation gu, this el-
ement transforms as gu→v 7→g−1
u gu→vgv, and the ﬁeld itself transforms as
x(v) 7→ρ(gv)x(v). If the ﬁlter commutes with the structure group represen-
tation Θ(u, v)ρ(gu) = ρ(gu)Θ(u, v), equation (23) deﬁnes a gauge-equivariant
convolution, which transforms as
(x′ ⋆Θ)(u) = ρ−1(gu)(x ⋆Θ)(u).


--- Page 65 ---
4. GEOMETRIC DOMAINS: THE 5 GS
61
under the aforementioned transformation.
4.6
Geometric graphs and Meshes
We will conclude our discussion of diﬀerent geometric domains with geo-
metric graphs (i.e., graphs that can be realised in some geometric space) and
meshes. In our ‘5G’ of geometric domains, meshes fall somewhere between
graphs and manifolds: in many regards, they are similar to graphs, but their
additional structure allows to also treat them similarly to continuous objects.
For this reason, we do not consider meshes as a standalone object in our
scheme, and in fact, will emphasise that many of the constructions we derive
in this section for meshes are directly applicable to general graphs as well.
As we already mentioned in Section 4.4, two-dimensional manifolds (sur-
faces) are a common way of modelling 3D objects (or, better said, the bound-
ary surfaces of such objects). In computer graphics and vision applications,
such surfaces are often discretised as triangular meshes,
Triangular meshes are
examples of topological
structures known as simplicial
complexes.
which can be roughly
thought of as a piece-wise planar approximation of a surface obtained by
gluing triangles together along their edges. Meshes are thus (undirected)
graphs with additional structure: in addition to nodes and edges, a mesh
T = (V, E, F) also have ordered triplets of nodes forming triangular faces
F = {(u, v, q) : u, v, q ∈V and (u, v), (u, q), (q, v) ∈E}; the order of the
nodes deﬁnes the face orientation.
Examples of manifold (top)
and non-manifold (bottom)
edges and nodes. For
manifolds with boundary,
one further deﬁnes boundary
edges that belong to exactly
one triangle.
It is further assumed that that each edge is shared by exactly two triangles,
and the boundary of all triangles incident on each node forms a single loop
of edges. This condition guarantees that 1-hop neighbourhoods around
each node are disk-like and the mesh thus constitutes a discrete manifold
– such meshes are referred to as manifold meshes. Similarly to Riemannian
manifolds, we can deﬁne a metric on the mesh. In the simplest instance, it can
be induced from the embedding of the mesh nodes x1, . . . , xn and expressed
through the Euclidean length of the edges, ℓuv = ∥xu−xv∥. A metric deﬁned
in this way automatically satisﬁes properties such as the triangle inequality,
i.e., expressions of the form ℓuv ≤ℓuq + ℓvq for any (u, v, q) ∈F and any
combination of edges. Any property that can be expressed solely in terms
of ℓis intrinsic, and any deformation of the mesh preserving ℓis an isometry
– these notions are already familiar to the reader from our discussion in
Section 4.4.


--- Page 66 ---
62
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
Laplacian matrices
By analogy to our treatment of graphs, let us assume a
(manifold) mesh with n nodes, each associated with a d-dimensional feature
vector, which we can arrange (assuming some arbitrary ordering) into an
n × d matrix X. The features can represent the geometric coordinates of
the nodes as well as additional properties such as colors, normals, etc, or
in speciﬁc applications such as chemistry where geometric graphs model
molecules, properties such as the atomic number.
Let us ﬁrst look at the spectral convolution (17) on meshes, which we remind
the readers, arises from the Laplacian operator. Considering the mesh as
a discretisation of an underlying continuous surface, we can discretise the
Laplacian as
(∆X)u =
X
v∈Nu
wuv(xu −xv),
(24)
or in matrix-vector notation, as an n × n symmetric matrix ∆= D −W,
where D = diag(d1, . . . , dn) is called the degree matrix and du = P
v wuv
the degree of node u. It is easy to see that equation (24) performs local
permutation-invariant aggregation of neighbour features φ(xu, XNu) =
duxu −P
v∈Nu wuvxv, and F(X) = ∆X is in fact an instance of our gen-
eral blueprint (13) for constructing permutation-equivariant functions on
graphs.
Note that insofar there is nothing speciﬁc to meshes in our deﬁnition of Lapla-
cian in (24); in fact, this construction is valid for arbitrary graphs as well,
with edge weights identiﬁed with the adjacency matrix, W = A, i.e., wuv = 1
The degree in this case equals
the number of neighbours.
if (u, v) ∈E and zero otherwise. Laplacians constructed in this way are often
called combinatorial, to reﬂect the fact that they merely capture the connectiv-
ity structure of the graph.
If the graph is directed, the
corresponding Laplacian is
non-symmetric.
For geometric graphs (which do not necessarily
have the additional structure of meshes, but whose nodes do have spatial
coordinates that induces a metric in the form of edge lengths), it is common
to use weights inversely related to the metric, e.g. wuv ∝e−ℓuv.
On meshes, we can exploit the additional structure aﬀorded by the faces,
and deﬁne the edge weights in equation (24) using the cotangent formula
(Pinkall and Polthier, 1993; Meyer et al., 2003)
The earliest use of this
formula dates back to the
PhD thesis of MacNeal
(1949), who developed it to
solve PDEs on the Caltech
Electric Analog Computer.
wuv = cot ∠uqv + cot ∠upv
2au
(25)
where ∠uqv and ∠upv are the two angles in the triangles (u, q, v) and (u, p, v)
opposite the shared edge (u, v), and au is the local area element, typically


--- Page 67 ---
4. GEOMETRIC DOMAINS: THE 5 GS
63
computed as the area of the polygon constructed upon the barycenters of the
triangles (u, p, q) sharing the node u and given by au = 1
3
P
v,q:(u,v,q)∈F auvq.
The cotangent Laplacian can be shown to have multiple convenient prop-
erties (see e.g. Wardetzky et al. (2007)): it is a positive-semideﬁnite matrix,
∆≽0 and thus has non-negative eigenvalues λ1 ≤. . . ≤λn that can be
regarded as an analogy of frequency, it is symmetric and thus has orthogonal
eigenvectors, and it is local (i.e., the value of (∆X)u depends only on 1-hop
neighbours, Nu). Perhaps the most important property is the convergence
of the cotangent mesh Laplacian matrix ∆to the continuous operator ∆
when the mesh is inﬁnitely reﬁned (Wardetzky, 2008). Equation (25) consti-
tutes thus an appropriate discretisation
Some technical conditions
must be imposed on the
reﬁnement, to avoid e.g.
triangles becoming
pathological. One such
example is a bizarre
triangulation of the cylinder
known in German as the
Schwarzscher Stiefel
(Schwarz’s boot) or in
English literature as the
‘Schwarz lantern’, proposed
in 1880 by Hermann Schwarz,
a German mathematician
known from the
Cauchy-Schwarz inequality
fame.
of the Laplacian operator deﬁned on
Riemannian manifolds in Section 4.4.
While one expects the Laplacian to be intrinsic, this is not very obvious from
equation (25), and it takes some eﬀort to express the cotangent weights
entirely in terms of the discrete metric ℓas
wuv = −ℓ2
uv + ℓ2
vq + ℓ2
uq
8auvq
+ −ℓ2
uv + ℓ2
vp + ℓ2
up
8auvp
where the area of the triangles aijk is given as
auvq =
q
suvq(suvq −ℓuv)(suvq −ℓvq)(suvq −ℓuq)
using Heron’s semiperimeter formula with suvq =
1
2(ℓuv + ℓuq + ℓvq). This
endows the Laplacian (and any quantities associated with it, such as its
eigenvectors and eigenvalues) with isometry invariance, a property for which
it is so loved in geometry processing and computer graphics (see an excellent
review by Wang and Solomon (2019)): any deformation of the mesh that
does not aﬀect the metric ℓ(does not ‘stretch’ or ‘squeeze’ the edges of the
mesh) does not change the Laplacian.
Finally, as we already noticed,
Laplacian-based ﬁlters are
isotropic. In the plane, such
ﬁlters have radial symmetry.
the deﬁnition of the Laplacian (25) is invariant
to the permutation of nodes in Nu, as it involves aggregation in the form
of summation. While on general graphs this is a necessary evil due to the
lack of canonical ordering of neighbours, on meshes we can order the 1-hop
neighbours according to some orientation (e.g., clock-wise), and the only
ambiguity is the selection of the ﬁrst node. Thus, instead of any possible
permutation we need to account for cyclic shifts (rotations), which intuitively
corresponds to the ambiguity arising from SO(2) gauge transformations


--- Page 68 ---
64
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
discussed in Section 4.5. For a ﬁxed gauge, it is possible to deﬁne an anisotropic
Laplacian that is sensitive to local directions and amounts to changing the
metric or the weights wuv. Constructions of this kind were used to design
shape descriptors by Andreux et al. (2014); Boscaini et al. (2016b) and in
early Geometric Deep Learning architectures on meshes by Boscaini et al.
(2016a).
Spectral analysis on meshes
The orthogonal eigenvectors Φ = (ϕ1, . . . , ϕn)
diagonalising the Laplacian matrix (∆= ΦΛΦ⊤, where Λ = diag(λ1, . . . , λn)
is the diagonal matrix of Laplacian eigenvalues), are used as the non-Euclidean
analogy of the Fourier basis, allowing to perform spectral convolution on
the mesh as the product of the respective Fourier transforms,
X ⋆θ = Φ diag(Φ⊤θ)(Φ⊤X) = Φ diag(ˆθ) ˆX,
where the ﬁlter ˆθ is designed directly in the Fourier domain. Again, nothing
in this formula is speciﬁc to meshes, and one can use the Laplacian matrix of
a generic (undirected) graph.
The fact that the graph is
assumed to be undirected is
important: in this case the
Laplacian is symmetric and
has orthogonal eigenvectors.
It is tempting to exploit this spectral deﬁnition
of convolution to generalise CNNs to graphs, which in fact was done by
one of the authors of this text, Bruna et al. (2013). However, it appears that
the non-Euclidean Fourier transform is extremely sensitive to even minor
perturbations of the underlying mesh or graph (see Figure 12 in Section 4.4)
and thus can only be used when one has to deal with diﬀerent signals on a
ﬁxed domain, but not when one wishes to generalise across diﬀerent domains.
Unluckily, many computer graphics and vision problems fall into the latter
category, where one trains a neural network on one set of 3D shapes (meshes)
and test on a diﬀerent set, making the Fourier transform-based approach
inappropriate.
As noted in Section 4.4, it is preferable to use spectral ﬁlters of the form (18)
applying some transfer function ˆp(λ) to the Laplacian matrix,
ˆp(∆)X = Φˆp(Λ)Φ⊤X = Φ diag(ˆp(λ1), . . . , ˆp(λn)) ˆX.
When ˆp can be expressed in terms of matrix-vector products, the eigende-
composition of the n × n matrix ∆
In the general case, the
complexity of
eigendecomposition is O(n3).
can be avoided altogether. For example,
Deﬀerrard et al. (2016) used polynomials of degree r as ﬁlter functions,
ˆp(∆)X =
r
X
k=0
αk∆kX = α0X + α1∆X + . . . + αr∆rX,


--- Page 69 ---
4. GEOMETRIC DOMAINS: THE 5 GS
65
amounting to the multiplication of the n × d feature matrix X by the n × n
Laplacian matrix r times. Since the Laplacian is typically sparse (with O(|E|)
non-zero elements)
Meshes are nearly-regular
graphs, with each node
having O(1) neighbours,
resulting in O(n) non-zeros
in ∆.
this operation has low complexity of O(|E|dr) ∼O(|E|).
Furthermore, since the Laplacian is local, a polynomial ﬁlter of degree r is
localised in r-hop neighbourhood.
However, this exact property comes at a disadvantage when dealing with
meshes, since the actual support of the ﬁlter (i.e., the radius it covers) de-
pends on the resolution of the mesh. One has to bear in mind that meshes
arise from the discretisation of some underlying continuous surface, and
one may have two diﬀerent meshes T and T ′ representing the same object.
Two-hop neighbourhoods on
meshes of diﬀerent
resolution.
In a ﬁner mesh, one might have to use larger neighbourhoods (thus, larger
degree r of the ﬁlter) than in a coarser one.
For this reason, in computer graphics applications it is more common to use
rational ﬁlters, since they are resolution-independent. There are many ways
to deﬁne such ﬁlters (see, e.g. Patanè (2020)), the most common being as a
polynomial of some rational function, e.g., λ−1
λ+1. More generally, one can use
a complex function, such as the Cayley transform λ−i
λ+i that maps the real line
into the unit circle in the complex plane.
Cayley transform is a
particular case of a Möbius
transformation. When applied
to the Laplacian (a
positive-semindeﬁnite
matrix), it maps its
non-negative eigenvalues to
the complex half-circle.
Levie et al. (2018) used spectral
ﬁlters expressed as Cayley polynomials, real rational functions with complex
coeﬃcients αl ∈C,
ˆp(λ) = Re
 r
X
l=0
αl
λ −i
λ + i
l!
.
When applied to matrices, the computation of the Cayley polynomial requires
matrix inversion,
ˆp(∆) = Re
 r
X
l=0
αl(∆−iI)l(∆+ iI)−l
!
,
In signal processing,
polynomial ﬁlters are termed
ﬁnite impulse response (FIR),
whereas rational ﬁlters are
inﬁnite impulse response (IIR).
which can be carried out approximately with linear complexity. Unlike
polynomial ﬁlters, rational ﬁlters do not have a local support, but have
exponential decay (Levie et al., 2018). A crucial diﬀerence compared to the
direct computation of the Fourier transform is that polynomial and rational
ﬁlters are stable under approximate isometric deformations of the underlying
graph or mesh – various results of this kind were shown e.g. by Levie et al.
(2018, 2019); Gama et al. (2020); Kenlay et al. (2021).


--- Page 70 ---
66
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
Meshes as operators and Functional maps
The paradigm of functional
maps suggests thinking of meshes as operators. As we will show, this allows
obtaining more interesting types of invariance exploiting the additional
structure of meshes. For the purpose of our discussion, assume the mesh T
is constructed upon embedded nodes with coordinates X. If we construct
an intrinsic operator like the Laplacian, it can be shown that it encodes
completely the structure of the mesh, and one can recover the mesh (up to its
isometric embedding, as shown by Zeng et al. (2012)). This is also true for
some other operators (see e.g. Boscaini et al. (2015); Corman et al. (2017);
Chern et al. (2018)), so we will assume a general operator, or n × n matrix
Q(T , X), as a representation of our mesh.
In this view, the discussion of Section 4.1 of learning functions of the form
f(X, T ) can be rephrased as learning functions of the form f(Q). Similar to
graphs and sets, the nodes of meshes also have no canonical ordering, i.e.,
functions on meshes must satisfy the permutation invariance or equivariance
conditions,
f(Q)
=
f(PQP⊤)
PF(Q)
=
F(PQP⊤)
for any permutation matrix P. However, compared to general graphs we
now have more structure: we can assume that our mesh arises from the
discretisation of some underlying continuous surface Ω. It is thus possible
to have a diﬀerent mesh T ′ = (V′, E′, F′) with n′ nodes and coordinates X′
representing the same object Ωas T . Importantly, the meshes T and T ′ can
have a diﬀerent connectivity structure and even diﬀerent number of nodes
(n′ ̸= n). Therefore, we cannot think of these meshes as isomorphic graphs
with mere reordering of nodes and consider the permutation matrix P as
correspondence between them.
Functional maps were introduced by Ovsjanikov et al. (2012) as a gener-
alisation of the notion of correspondence to such settings, replacing the
correspondence between points on two domains (a map η : Ω→Ω′) with
correspondence between functions (a map C : X(Ω) →X(Ω′), see Figure 13).
A functional map is a linear operator C, represented as a matrix n′ × n, estab-
lishing correspondence between signals x′ and x on the respective domains
as
x′ = Cx.
In most cases the functional
map is implemented in the
spectral domain, as a k × k
map ˆC between the Fourier
coeﬃcients, x′ = Φ′ ˆCΦ⊤x,
where Φ and Φ′ are the
respective n × k and n′ × k
matrices of the (truncated)
Laplacian eigenbases, with
k ≪n, n′.
Rustamov et al. (2013) showed that in order to guarantee area-preserving
mapping, the functional map must be orthogonal, C⊤C = I, i.e., be an


--- Page 71 ---
4. GEOMETRIC DOMAINS: THE 5 GS
67
element of the orthogonal group C ∈O(n). In this case, we can invert the
map using C−1 = C⊤.
Figure 13: Pointwise map (left) vs functional map (right).
The functional map also establishes a relation between the operator repre-
sentation of meshes,
Q′ = CQC⊤,
Q = C⊤Q′C,
which we can interpret as follows: given an operator representation Q of T
and a functional map C, we can construct its representation Q′ of T ′ by ﬁrst
mapping the signal from T ′ to T (using C⊤), applying the operator Q, and
then mapping back to T ′ (using C)
Note that we read these
operations right-to-left.
This leads us to a more general class of
remeshing invariant (or equivariant) functions on meshes, satisfying
f(Q)
=
f(CQC⊤) = f(Q′)
CF(Q)
=
F(CQC⊤) = F(Q′)
for any C ∈O(n). It is easy to see that the previous setting of permutation
invariance and equivariance is a particular case,
This follows from the
orthogonality of permutation
matrices, P⊤P = I.
which can be thought of as
a trivial remeshing in which only the order of nodes is changed.
Wang et al. (2019a) showed that given an eigendecomposition of the opera-
tor Q = VΛV⊤, any remeshing invariant (or equivariant) function can
be expressed as f(Q) = f(Λ) and F(Q) = VF(Λ), or in other words,
remeshing-invariant functions involve only the spectrum of Q. Indeed, func-
tions of Laplacian eigenvalues have been proven in practice to be robust to
surface discretisation and perturbation, explaining the popularity of spectral
constructions based on Laplacians in computer graphics, as well as in deep
learning on graph (Deﬀerrard et al., 2016; Levie et al., 2018). Since this result
refers to a generic operator Q, multiple choices are available besides the
ubiquitous Laplacian – notable examples include the Dirac (Liu et al., 2017;
Kostrikov et al., 2018) or Steklov (Wang et al., 2018) operators, as well as
learnable parametric operators (Wang et al., 2019a).
