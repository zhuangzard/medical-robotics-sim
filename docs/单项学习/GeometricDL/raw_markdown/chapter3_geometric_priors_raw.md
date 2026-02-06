# 3 Geometric Priors

--- Page 13 ---
3. GEOMETRIC PRIORS
9
Figure 2: We consider a Lipschitz function f(x) = P2d
j=1 zjφ(x −xj) where
zj = ±1, xj ∈Rd is placed in each quadrant, and φ a locally supported
Lipschitz ‘bump’. Unless we observe the function in most of the 2d quad-
rants, we will incur in a constant error in predicting it. This simple geo-
metric argument can be formalised through the notion of Maximum Dis-
crepancy (von Luxburg and Bousquet, 2004), deﬁned for the Lipschitz class
as κ(d) = Ex,x′ supf∈Lip(1)
 1
N
P
l f(xl) −1
N
P
l f(x′
l)
 ≃N−1/d, which mea-
sures the largest expected discrepancy between two independent N-sample
expectations. Ensuring that κ(d) ≃ϵ requires N = Θ(ϵ−d); the correspond-
ing sample {xl}l deﬁnes an ϵ-net of the domain. For a d-dimensional Eu-
clidean domain of diameter 1, its size grows exponentially as ϵ−d.
3
Geometric Priors
Modern data analysis is synonymous with high-dimensional learning. While
the simple arguments of Section 2.1 reveal the impossibility of learning from
generic high-dimensional data as a result of the curse of dimensionality,
there is hope for physically-structured data, where we can employ two fun-
damental principles: symmetry and scale separation. In the settings considered
in this text, this additional structure will usually come from the structure of
the domain underlying the input signals: we will assume that our machine
learning system operates on signals (functions) on some domain Ω. While
in many cases linear combinations of points on Ωis not well-deﬁned
Ωmust be a vector space in
order for an expression
αu + βv to make sense.
, we can
linearly combine signals on it, i.e., the space of signals forms a vector space.
Moreover, since we can deﬁne an inner product between signals, this space
is a Hilbert space.


--- Page 14 ---
10
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
Figure 3: If the unknown function f is presumed to be well approximated as
f(x) ≈g(Ax) for some unknown A ∈Rk×d with k ≪d, then shallow neural
networks can capture this inductive bias, see e.g. Bach (2017). In typical
applications, such dependency on low-dimensional projections is unrealistic,
as illustrated in this example: a low-pass ﬁlter projects the input images
to a low-dimensional subspace; while it conveys most of the semantics,
substantial information is lost.


--- Page 15 ---
3. GEOMETRIC PRIORS
11
The space of C-valued signals on Ω
When Ωhas some additional
structure, we may further
restrict the kinds of signals in
X(Ω, C). For example, when
Ωis a smooth manifold, we
may require the signals to be
smooth. Whenever possible,
we will omit the range C for
brevity.
(for Ωa set, possibly with additional
structure, and C a vector space, whose dimensions are called channels)
X(Ω, C) = {x : Ω→C}
(1)
is a function space that has a vector space structure. Addition and scalar
multiplication of signals is deﬁned as:
(αx + βy)(u) = αx(u) + βy(u)
for all
u ∈Ω,
with real scalars α, β. Given an inner product ⟨v, w⟩C on C and a measure
When the domain Ωis
discrete, µ can be chosen as
the counting measure, in which
case the integral becomes a
sum. In the following, we
will omit the measure and
use du for brevity.
µ on Ω(with respect to which we can deﬁne an integral), we can deﬁne
an inner product on X(Ω, C) as
⟨x, y⟩=
Z
Ω
⟨x(u), y(u)⟩C dµ(u).
(2)
As a typical illustration, take Ω= Zn ×Zn to be a two-dimensional n×n grid,
x an RGB image (i.e. a signal x : Ω→R3), and f a function (such as a single-
layer Perceptron) operating on 3n2-dimensional inputs. As we will see in the
following with greater detail, the domain Ωis usually endowed with certain
geometric structure and symmetries. Scale separation results from our ability
to preserve important characteristics of the signal when transferring it onto
a coarser version of the domain (in our example, subsampling the image by
coarsening the underlying grid).
We will show that both principles, to which we will generically refer as geo-
metric priors, are prominent in most modern deep learning architectures. In
the case of images considered above, geometric priors are built into Convo-
lutional Neural Networks (CNNs) in the form of convolutional ﬁlters with
shared weights (exploiting translational symmetry) and pooling (exploiting
scale separation). Extending these ideas to other domains such as graphs
and manifolds and showing how geometric priors emerge from fundamental
principles is the main goal of Geometric Deep Learning and the leitmotif of
our text.


--- Page 16 ---
12
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
3.1
Symmetries, Representations, and Invariance
Informally, a symmetry of an object or system is a transformation that leaves
a certain property of said object or system unchanged or invariant. Such
transformations may be either smooth, continuous, or discrete. Symmetries
are ubiquitous in many machine learning tasks. For example, in computer
vision the object category is unchanged by shifts, so shifts are symmetries in
the problem of visual object classiﬁcation. In computational chemistry, the
task of predicting properties of molecules independently of their orientation
in space requires rotational invariance. Discrete symmetries emerge naturally
when describing particle systems where particles do not have canonical
ordering and thus can be arbitrarily permuted, as well as in many dynamical
systems, via the time-reversal symmetry (such as systems in detailed bal-
ance or the Newton’s second law of motion). As we will see in Section 4.1,
permutation symmetries are also central to the analysis of graph-structured
data.
Symmetry groups
The set of symmetries of an object satisﬁes a number of
properties. First, symmetries may be combined to obtain new symmetries:
if g and h are two symmetries, then their compositions g ◦h and h ◦g
We will follow the
juxtaposition notation
convention used in group
theory, g ◦h = gh, which
should be read right-to-left:
we ﬁrst apply h and then g.
The order is important, as in
many cases symmetries are
non-commutative. Readers
familiar with Lie groups
might be disturbed by our
choice to use the Fraktur font
to denote group elements, as
it is a common notation of Lie
algebras.
are
also symmetries. The reason is that if both transformations leave the object
invariant, then so does the composition of transformations, and hence the
composition is also a symmetry. Furthermore, symmetries are always in-
vertible, and the inverse is also a symmetry. This shows that the collection
of all symmetries form an algebraic object known as a group. Since these
objects will be a centerpiece of the mathematical model of Geometric Deep
Learning, they deserve a formal deﬁnition and detailed discussion:


--- Page 17 ---
3. GEOMETRIC PRIORS
13
A group is a set G along with a binary operation ◦: G × G →G called
composition (for brevity, denoted by juxtaposition g ◦h = gh) satisfying
the following axioms:
Associativity: (gh)k = g(hk) for all g, h, k ∈G.
Identity: there exists a unique e ∈G satisfying eg = ge = g for all g ∈G.
Inverse: For each g ∈G there is a unique inverse g−1 ∈G such that
gg−1 = g−1g = e.
Closure: The group is closed under composition, i.e., for every g, h ∈G,
we have gh ∈G.
Note that commutativity is not part of this deﬁnition, i.e. we may have gh ̸= hg.
Groups for which gh = hg for all g, h ∈G are called commutative or Abelian
After the Norwegian
mathematician Niels Henrik
Abel (1802–1829).
.
Though some groups can be very large and even inﬁnite, they often arise
from compositions of just a few elements, called group generators. Formally,
G is said to be generated by a subset S ⊆G (called the group generator) if
every element g ∈G can be written as a ﬁnite composition of the elements
of S and their inverses. For instance, the symmetry group of an equilateral
triangle (dihedral group D3) is generated by a 60◦rotation and a reﬂection
(Figure 4). The 1D translation group, which we will discuss in detail in the
following, is generated by inﬁnitesimal displacements; this is an example of
a Lie group of diﬀerentiable symmetries.
Lie groups have a
diﬀerentiable manifold
structure. One such example
that we will study in
Section 4.3 is the special
orthogonal group SO(3),
which is a 3-dimensional
manifold.
Note that here we have deﬁned a group as an abstract object, without saying
what the group elements are (e.g. transformations of some domain), only
how they compose. Hence, very diﬀerent kinds of objects may have the same
symmetry group. For instance, the aforementioned group of rotational and
reﬂection symmetries of a triangle is the same as the group of permutations
of a sequence of three elements (we can permute the corners in the triangle
in any way using a rotation and reﬂection – see Figure 4)
The diagram shown in Figure
4 (where each node is
associated with a group
element, and each arrow with
a generator), is known as the
Cayley diagram.
.
Group Actions and Group Representations
Rather than considering groups
as abstract entities, we are mostly interested in how groups act on data. Since
we assumed that there is some domain Ωunderlying our data, we will study
how the group acts on Ω(e.g. translation of points of the plane), and from
there obtain actions of the same group on the space of signals X(Ω) (e.g.
translations of planar images and feature maps).


--- Page 18 ---
14
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
Figure 4: Left: an equilateral triangle with corners labelled by 1, 2, 3, and
all possible rotations and reﬂections of the triangle. The group D3 of rota-
tion/reﬂection symmetries of the triangle is generated by only two elements
(rotation by 60◦R and reﬂection F) and is the same as the group Σ3 of per-
mutations of three elements. Right: the multiplication table of the group D3.
The element in the row g and column h corresponds to the element gh.
A group action
Technically, what we deﬁne
here is a left group action.
of G on a set Ωis deﬁned as a mapping (g, u) 7→g.u associating
a group element g ∈G and a point u ∈Ωwith some other point on Ωin a
way that is compatible with the group operations, i.e., g.(h.u) = (gh).u for
all g, h ∈G and u ∈Ω. We shall see numerous instances of group actions in
the following sections. For example, in the plane the Euclidean group E(2) is
the group of transformations of R2 that preserves Euclidean distances
Distance-preserving
transformations are called
isometries. According to
Klein’s Erlangen Programme,
the classical Euclidean
geometry arises from this
group.
, and
consists of translations, rotations, and reﬂections. The same group, however,
can also act on the space of images on the plane (by translating, rotating and
ﬂipping the grid of pixels), as well as on the representation spaces learned
by a neural network. More precisely, if we have a group G acting on Ω, we
automatically obtain an action of G on the space X(Ω):
(g.x)(u) = x(g−1u).
(3)
Due to the inverse on g, this is indeed a valid group action, in that we have
(g.(h.x))(u) = ((gh).x)(u).
The most important kind of group actions, which we will encounter repeat-
edly throughout this text, are linear group actions, also known as group
representations. The action on signals in equation (3) is indeed linear, in the


--- Page 19 ---
3. GEOMETRIC PRIORS
15
sense that
g.(αx + βx′) = α(g.x) + β(g.x′)
for any scalars α, β and signals x, x′ ∈X(Ω). We can describe linear actions
either as maps (g, x) 7→g.x that are linear in x, or equivalently, by currying,
as a map ρ : G →Rn×n
When Ωis inﬁnte, the space
of signals X(Ω) is inﬁnite
dimensional, in which case
ρ(g) is a linear operator on
this space, rather than a ﬁnite
dimensional matrix. In
practice, one must always
discretise to a ﬁnite grid,
though.
that assigns to each group element g an (invertible)
matrix ρ(g). The dimension n of the matrix is in general arbitrary and not
necessarily related to the dimensionality of the group or the dimensionality
of Ω, but in applications to deep learning n will usually be the dimensionality
of the feature space on which the group acts. For instance, we may have the
group of 2D translations acting on a space of images with n pixels.
As with a general group action, the assignment of matrices to group elements
should be compatible with the group action. More speciﬁcally, the matrix
representing a composite group element gh should equal the matrix product
of the representation of g and h:
A n-dimensional real representation of a group G is a map ρ : G →Rn×n,
assigning to each g ∈G an invertible matrix ρ(g), and satisfying the
condition ρ(gh) = ρ(g)ρ(h) for all g, h ∈G.
Similarly, a complex
representation is a map
ρ : G →Cn×n satisfying the
same equation.
A representation is called
unitary or orthogonal if the matrix ρ(g) is unitary or orthogonal for all
g ∈G.
Written in the language of group representations, the action of G on signals
x ∈X(Ω) is deﬁned as ρ(g)x(u) = x(g−1u). We again verify that
(ρ(g)(ρ(h)x))(u) = (ρ(gh)x)(u).
Invariant and Equivariant functions
The symmetry of the domain Ωun-
derlying the signals X(Ω) imposes structure on the function f deﬁned on
such signals. It turns out to be a powerful inductive bias, improving learning
In general, f depends both
on the signal an the domain,
i.e., F(X(Ω), Ω). We will
often omit the latter
dependency for brevity.
eﬃciency by reducing the space of possible interpolants, F(X(Ω)), to those
which satisfy the symmetry priors. Two important cases we will be exploring
in this text are invariant and equivariant functions.
A function f : X(Ω) →Y is G-invariant if f(ρ(g)x) = f(x) for all g ∈G
and x ∈X(Ω), i.e., its output is unaﬀected by the group action on the
input.


--- Page 20 ---
16
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
Figure 5: Three spaces of interest in Geometric Deep Learning: the (physi-
cal) domain Ω, the space of signals X(Ω), and the hypothesis class F(X(Ω)).
Symmetries of the domain Ω(captured by the group G) act on signals
x ∈X(Ω) through group representations ρ(g), imposing structure on the
functions f ∈F(X(Ω)) acting on such signals.
A classical example of invariance is shift-invariance,
Note that signal processing
books routinely use the term
‘shift-invariance’ referring to
shift-equivariance, e.g. Linear
Shift-invariant Systems.
arising in computer
vision and pattern recognition applications such as image classiﬁcation. The
function f in this case (typically implemented as a Convolutional Neural
Network) inputs an image and outputs the probability of the image to contain
an object from a certain class (e.g. cat or dog). It is often reasonably assumed
that the classiﬁcation result should not be aﬀected by the position of the
object in the image, i.e., the function f must be shift-invariant. Multi-layer
Perceptrons, which can approximate any smooth function, do not have this
property – one of the reasons why early attempts to apply these architectures
to problems of pattern recognition in the 1970s failed. The development of
neural network architectures with local weight sharing, as epitomised by
Convolutional Neural Networks, was, among other reasons, motivated by
the need for shift-invariant object classiﬁcation.
If we however take a closer look at the convolutional layers of CNNs, we
will ﬁnd that they are not shift-invariant but shift-equivariant: in other words,
a shift of the input to a convolutional layer produces a shift in the output
feature maps by the same amount.
A function f : X(Ω) →X(Ω) is G-equivariant if
More generally, we might
have f : X(Ω) →X(Ω′) with
input and output spaces
having diﬀerent domains
Ω, Ω′ and representations ρ,
ρ′ of the same group G. In
this case, equivariance is
deﬁned as
f(ρ(g)x) = ρ′(g)f(x).
f(ρ(g)x) = ρ(g)f(x) for
all g ∈G, i.e., group action on the input aﬀects the output in the same
way.
Resorting again to computer vision, a prototypical application requiring


--- Page 21 ---
3. GEOMETRIC PRIORS
17
shift-equivariance is image segmentation, where the output of f is a pixel-
wise image mask. Obviously, the segmentation mask must follow shifts in
the input image. In this example, the domains of the input and output are
the same, but since the input has three color channels while the output has
one channel per class, the representations (ρ, X(Ω, C)) and (ρ′, X(Ω, C′)) are
somewhat diﬀerent.
However, even the previous use case of image classiﬁcation is usually imple-
mented as a sequence of convolutional (shift-equivariant) layers, followed by
global pooling (which is shift-invariant). As we will see in Section 3.5, this
is a general blueprint of a majority of deep learning architectures, including
CNNs and Graph Neural Networks (GNNs).
3.2
Isomorphisms and Automorphisms
Subgroups and Levels of structure
As mentioned before, a symmetry
Invertible and
structure-preserving maps
between diﬀerent objects
often go under the generic
name of isomorphisms (Greek
for ‘equal shape’). An
isomorphism from an object
to itself is called an
automorphism, or symmetry.
is
a transformation that preserves some property or structure, and the set of
all such transformations for a given structure forms a symmetry group. It
happens often that there is not one but multiple structures of interest, and
so we can consider several levels of structure on our domain Ω. Hence, what
counts as a symmetry depends on the structure under consideration, but in
all cases a symmetry is an invertible map that respects this structure.
On the most basic level, the domain Ωis a set, which has a minimal amount
of structure: all we can say is that the set has some cardinality
For a ﬁnite set, the cardinality
is the number of elements
(‘size’) of the set, and for
inﬁnite sets the cardinality
indicates diﬀerent kinds of
inﬁnities, such as the
countable inﬁnity of the
natural numbers, or the
uncountable inﬁnity of the
continuum R.
. Self-maps
that preserve this structure are bijections (invertible maps), which we may
consider as set-level symmetries. One can easily verify that this is a group
by checking the axioms: a compositions of two bijections is also a bijection
(closure), the associativity stems from the associativity of the function com-
position, the map τ(u) = u is the identity element, and for every τ the inverse
exists by deﬁnition, satisfying (τ ◦τ −1)(u) = (τ −1 ◦τ)(u) = u.
Depending on the application, there may be further levels of structure. For
instance, if Ωis a topological space, we can consider maps that preserve
continuity: such maps are called homeomorphisms and in addition to simple
bijections between sets, are also continuous and have continuous inverse.
Intuitively, continuous functions are well-behaved and map points in a neigh-
bourhood (open set) around a point u to a neighbourhood around τ(u).


--- Page 22 ---
18
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
One can further demand that the map and its inverse are (continuously)
diﬀerentiable,
Every diﬀerentiable function
is continuous. If the map is
continuously diﬀerentiable
‘suﬃciently many times’, it is
said to be smooth.
i.e., the map and its inverse have a derivative at every point
(and the derivative is also continuous). This requires further diﬀerentiable
structure that comes with diﬀerentiable manifolds, where such maps are
called diﬀeomorphisms and denoted by Diﬀ(Ω). Additional examples of struc-
tures we will encounter include distances or metrics (maps preserving them
are called isometries) or orientation (to the best of our knowledge, orientation-
preserving maps do not have a common Greek name).
A metric or distance is a function d : Ω× Ω→[0, ∞) satisfying for all
u, v, w ∈Ω:
Identity of indiscernibles: d(u, v) = 0 iﬀu = v.
Symmetry: d(u, v) = d(v, u).
Triangle inequality: d(u, v) ≤d(u, w) + d(w, v).
A space equipped with a metric (Ω, d) is called a metric space.
The right level of structure to consider depends on the problem. For example,
when segmenting histopathology slide images, we may wish to consider
ﬂipped versions of an image as equivalent (as the sample can be ﬂipped
when put under the microscope), but if we are trying to classify road signs,
we would only want to consider orientation-preserving transformations as
symmetries (since reﬂections could change the meaning of the sign).
As we add levels of structure to be preserved, the symmetry group will get
smaller. Indeed, adding structure is equivalent to selecting a subgroup, which
is a subset of the larger group that satisﬁes the axioms of a group by itself:
Let (G, ◦) be a group and H ⊆G a subset. H is said to be a subgroup of G
if (H, ◦) constitutes a group with the same operation.
For instance, the group of Euclidean isometries E(2) is a subgroup of the
group of planar diﬀeomorphisms Diﬀ(2), and in turn the group of orientation-
preserving isometries SE(2) is a subgroup of E(2). This hierarchy of struc-
ture follows the Erlangen Programme philosophy outlined in the Preface:
in Klein’s construction, the Projective, Aﬃne, and Euclidean geometries


--- Page 23 ---
3. GEOMETRIC PRIORS
19
have increasingly more invariants and correspond to progressively smaller
groups.
Isomorphisms and Automorphisms
We have described symmetries as
structure preserving and invertible maps from an object to itself. Such maps
are also known as automorphisms, and describe a way in which an object
is equivalent it itself. However, an equally important class of maps are
the so-called isomorphisms, which exhibit an equivalence between two non-
identical objects. These concepts are often conﬂated, but distinguishing them
is necessary to create clarity for our following discussion.
To understand the diﬀerence, consider a set Ω= {0, 1, 2}. An automorphism
of the set Ωis a bijection τ : Ω→Ωsuch as a cyclic shift τ(u) = u + 1 mod 3.
Such a map preserves the cardinality property, and maps Ωonto itself. If
we have another set Ω′ = {a, b, c} with the same number of elements, then a
bijection η : Ω→Ω′ such as η(0) = a, η(1) = b, η(2) = c is a set isomorphism.
As we will see in Section 4.1 for graphs, the notion of structure includes
not just the number of nodes, but also the connectivity. An isomorphism
η : V →V′ between two graphs G = (V, E) and G′ = (V′, E′) is thus a
bijection between the nodes that maps pairs of connected nodes to pairs
of connected nodes, and likewise for pairs of non-connected nodes.
I.e., (η(u), η(v)) ∈V′ iﬀ
(u, v) ∈V.
Two
isomorphic graphs are thus structurally identical, and diﬀer only in the
way their nodes are ordered.
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
1
3
5
7
9
11
13
15
17
19
The Folkman graph (Folkman,
1967) is a beautiful example
of a graph with 3840
automorphisms, exempliﬁed
by the many symmetric ways
to draw it.
On the other hand, a graph automorphism or
symmetry is a map τ : V →V maps the nodes of the graph back to itself,
while preserving the connectivity. A graph with a non-trivial automorphism
(i.e., τ ̸= id) presents symmetries.
3.3
Deformation Stability
The symmetry formalism introduced in Sections 3.1–3.2 captures an idealised
world where we know exactly which transformations are to be considered as
symmetries, and we want to respect these symmetries exactly. For instance
in computer vision, we might assume that planar translations are exact
symmetries. However, the real world is noisy and this model falls short in
two ways.
Two objects moving at
diﬀerent velocities in a video
deﬁne a transformation
outside the translation group.
Firstly, while these simple groups provide a way to understand global sym-


--- Page 24 ---
20
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
metries of the domain Ω(and by extension, of signals on it, X(Ω)), they do
not capture local symmetries well. For instance, consider a video scene with
several objects, each moving along its own diﬀerent direction. At subsequent
frames, the resulting scene will contain approximately the same semantic
information, yet no global translation explains the transformation from one
frame to another. In other cases, such as a deformable 3D object viewed by
a camera, it is simply very hard to describe the group of transformations
that preserve the object identity. These examples illustrate that in reality
we are more interested in a far larger set of transformations where global,
exact invariance is replaced by a local, inexact one. In our discussion, we
will distinguish between two scenarios: the setting where the domain Ωis
ﬁxed, and signals x ∈X(Ω) are undergoing deformations, and the setting
where the domain Ωitself may be deformed.
Stability to signal deformations
In many applications, we know a priori
that a small deformation of the signal x should not change the output of
f(x), so it is tempting to consider such deformations as symmetries. For
instance, we could view small diﬀeomorphisms τ ∈Diﬀ(Ω), or even small
bijections, as symmetries. However, small deformations can be composed
to form large deformations, so “small deformations” do not form a group,
E.g., the composition of two
ϵ-isometries is a 2ϵ-isometry,
violating the closure property.
and we cannot ask for invariance or equivariance to small deformations only.
Since large deformations can can actually materially change the semantic
content of the input, it is not a good idea to use the full group Diﬀ(Ω) as
symmetry group either.
A better approach is to quantify how “far” a given τ ∈Diﬀ(Ω) is from a
given symmetry subgroup G ⊂Diﬀ(Ω) (e.g. translations) with a complexity
measure c(τ), so that c(τ) = 0 whenever τ ∈G. We can now replace our
previous deﬁnition of exact invariance and equivarance under group actions
with a ‘softer’ notion of deformation stability (or approximate invariance):
∥f(ρ(τ)x) −f(x)∥≤Cc(τ)∥x∥, , ∀x ∈X(Ω)
(4)
where ρ(τ)x(u) = x(τ −1u) as before, and where C is some constant indepen-
dent of the signal x. A function f ∈F(X(Ω)) satisfying the above equation
is said to be geometrically stable. We will see examples of such functions in
the next Section 3.4.
Since c(τ) = 0 for τ ∈G, this deﬁnition generalises the G-invariance prop-
erty deﬁned above. Its utility in applications depends on introducing an


--- Page 25 ---
3. GEOMETRIC PRIORS
21
appropriate deformation cost. In the case of images deﬁned over a contin-
uous Euclidean plane, a popular choice is c2(τ) :=
R
Ω∥∇τ(u)∥2du, which
measures the ‘elasticity’ of τ, i.e., how diﬀerent it is from the displacement
by a constant vector ﬁeld. This deformation cost is in fact a norm often
called the Dirichlet energy, and can be used to quantify how far τ is from the
translation group.
Figure 6: The set of all bijective mappings from Ωinto itself forms the set
automorphism group Aut(Ω), of which a symmetry group G (shown as a circle)
is a subgroup. Geometric Stability extends the notion of G-invariance and
equivariance to ‘transformations around G’ (shown as gray ring), quantiﬁed
in the sense of some metric between transformations. In this example, a
smooth distortion of the image is close to a shift.
Stability to domain deformations
In many applications, the object being
deformed is not the signal, but the geometric domain Ωitself. Canonical
instances of this are applications dealing with graphs and manifolds: a graph
can model a social network at diﬀerent instance of time containing slightly
diﬀerent social relations (follow graph), or a manifold can model a 3D object
undergoing non-rigid deformations. This deformation can be quantiﬁed


--- Page 26 ---
22
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
as follows. If D denotes the space of all possible variable domains (such
as the space of all graphs, or the space of Riemannian manifolds), one can
deﬁne for Ω, ˜Ω∈D an appropriate metric (‘distance’) d(Ω, ˜Ω) satisfying
d(Ω, ˜Ω) = 0 if Ωand ˜Ωare equivalent in some sense: for example, the graph
edit distance vanishes when the graphs are isomorphic, and the Gromov-
Hausdorﬀdistance between Riemannian manifolds equipped with geodesic
distances vanishes when two manifolds are isometric.
The graph edit distance
measures the minimal cost of
making two graphs
isomorphic by a sequences of
graph edit operations. The
Gromov-Hausdorﬀdistance
measures the smallest
possible metric distortion of a
correspondence between two
metric spaces, see Gromov
(1981).
A common construction of such distances between domains relies on some
family of invertible mapping η : Ω→˜Ωthat try to ‘align’ the domains in a
way that the corresponding structures are best preserved. For example, in
the case of graphs or Riemannian manifolds (regarded as metric spaces with
the geodesic distance), this alignment can compare pair-wise adjacency or
distance structures (d and ˜d, respectively),
dD(Ω, ˜Ω) = inf
η∈G ∥d −˜d ◦(η × η)∥
where G is the group of isomorphisms such as bijections or isometries,
and the norm is deﬁned over the product space Ω× Ω. In other words,
a distance between elements of Ω, ˜Ωis ‘lifted’ to a distance between the
domains themselves, by accounting for all the possible alignments that
preserve the internal structure.
Two graphs can be aligned by
the Quadratic Assignment
Problem (QAP), which
considers in its simplest form
two graphs G, ˜G of the same
size n, and solves
minP∈Σn trace(AP ˜AP
⊤),
where A, ˜A are the
respective adjacency matrices
and Σn is the group of n × n
permutation matrices. The
graph edit distance can be
associated with such QAP
(Bougleux et al., 2015).
Given a signal x ∈X(Ω) and a deformed
domain ˜Ω, one can then consider the deformed signal ˜x = x ◦η−1 ∈X(˜Ω).
By slightly abusing the notation, we deﬁne X(D) = {(X(Ω), Ω) : Ω∈D} as
the ensemble of possible input signals deﬁned over a varying domain. A
function f : X(D) →Y is stable to domain deformations if
∥f(x, Ω) −f(˜x, ˜Ω)∥≤C∥x∥dD(Ω, ˜Ω)
(5)
for all Ω, ˜Ω∈D, and x ∈X(Ω). We will discuss this notion of stability in the
context of manifolds in Sections 4.4–4.6, where isometric deformations play
a crucial role. Furthermore, it can be shown that the stability to domain de-
formations is a natural generalisation of the stability to signal deformations,
by viewing the latter in terms of deformations of the volume form Gama
et al. (2019).
3.4
Scale Separation
While deformation stability substantially strengthens the global symmetry
priors, it is not suﬃcient in itself to overcome the curse of dimensionality, in


--- Page 27 ---
3. GEOMETRIC PRIORS
23
the sense that, informally speaking, there are still “too many" functions that
respect (4) as the size of the domain grows. A key insight to overcome this
curse is to exploit the multiscale structure of physical tasks. Before describ-
ing multiscale representations, we need to introduce the main elements of
Fourier transforms, which rely on frequency rather than scale.
Fourier Transform and Global invariants
Arguably
Fourier basis functions have
global support. As a result,
local signals produce energy
across all frequencies.
the most famous sig-
nal decomposition is the Fourier transform, the cornerstone of harmonic
analysis. The classical one-dimensional Fourier transform
ˆx(ξ) =
Z +∞
−∞
x(u)e−iξudu
expresses the function x(u) ∈L2(Ω) on the domain Ω= R as a linear
combination of orthogonal oscillating basis functions ϕξ(u) = eiξu, indexed by
their rate of oscillation (or frequency) ξ. Such an organisation into frequencies
reveals important information about the signal, e.g. its smoothness and
localisation. The Fourier basis itself has a deep geometric foundation and
can be interpreted as the natural vibrations of the domain, related to its
geometric structure (see e.g. Berger (2012)).
The Fourier transform
In the following, we will use
convolution and
(cross-)correlation
(x ⋆θ)(u) =
Z +∞
−∞
x(v)θ(u+v)dv
interchangeably, as it is
common in machine learning:
the diﬀerence between the
two is whether the ﬁlter is
reﬂected, and since the ﬁlter
is typically learnable, the
distinction is purely
notational.
plays a crucial role in signal processing as it oﬀers a
dual formulation of convolution,
(x ⋆θ)(u) =
Z +∞
−∞
x(v)θ(u −v)dv
a standard model of linear signal ﬁltering (here and in the following, x
denotes the signal and θ the ﬁlter). As we will show in the following, the
convolution operator is diagonalised in the Fourier basis, making it possible
to express convolution as the product of the respective Fourier transforms,
\
(x ⋆θ)(ξ) = ˆx(ξ) · ˆθ(ξ),
a fact known in signal processing as the Convolution Theorem.
As it turns out, many fundamental diﬀerential operators such as the Lapla-
cian are described as convolutions on Euclidean domains. Since such diﬀer-
ential operators can be deﬁned intrinsically over very general geometries,
this provides a formal procedure to extend Fourier transforms beyond Eu-
clidean domains, including graphs, groups and manifolds. We will discuss
this in detail in Section 4.4.


--- Page 28 ---
24
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
An essential aspect of Fourier transforms is that they reveal global properties
of the signal and the domain, such as smoothness or conductance. Such
global behavior is convenient in presence of global symmetries of the domain
such as translation, but not to study more general diﬀeomorphisms. This
requires a representation that trades oﬀspatial and frequential localisation,
as we see next.
Multiscale representations
The notion of local invariance can be articu-
lated by switching from a Fourier frequency-based representation to a scale-
based representation, the cornerstone of multi-scale decomposition methods
such as wavelets.
See Mallat (1999) for a
comperehensive introduction.
The essential insight of multi-scale methods is to decom-
pose functions deﬁned over the domain Ωinto elementary functions that are
localised both in space and frequency.
Contrary to Fourier, wavelet
atoms are localised and
multi-scale, allowing to
capture ﬁne details of the
signal with atoms having
small spatial support and
coarse details with atoms
having large spatial support.
The term atom here is
synonymous with ‘basis
element’ in Fourier analysis,
with the caveat that wavelets
are redundant
(over-complete).
In the case of wavelets, this is achieved
by correlating a translated and dilated ﬁlter (mother wavelet) ψ, producing a
combined spatio-frequency representation called a continuous wavelet trans-
form
(Wψx)(u, ξ) = ξ−1/2
Z +∞
−∞
ψ
v −u
ξ

x(v)dv.
The translated and dilated ﬁlters are called wavelet atoms; their spatial po-
sition and dilation correspond to the coordinates u and ξ of the wavelet
transform. These coordinates are usually sampled dyadically (ξ = 2−j and
u = 2−jk), with j referred to as scale. Multi-scale signal representations
bring important beneﬁts in terms of capturing regularity properties beyond
global smoothness, such as piece-wise smoothness, which made them a
popular tool in signal and image processing and numerical analysis in the
90s.
Deformation stability of Multiscale representations:
The beneﬁt of mul-
tiscale localised wavelet decompositions over Fourier decompositions is
revealed when considering the eﬀect of small deformations ‘nearby’ the
underlying symmetry group. Let us illustrate this important concept in the
Euclidean domain and the translation group. Since the Fourier representa-
tion diagonalises the shift operator (which can be thought of as convolution,
as we will see in more detail in Section 4.2), it is an eﬃcient representation for
translation transformations. However, Fourier decompositions are unstable
under high-frequency deformations. In contrast, wavelet decompositions
oﬀer a stable representation in such cases.


--- Page 29 ---
3. GEOMETRIC PRIORS
25
Indeed, let us consider τ ∈Aut(Ω) and its associated linear representation
ρ(τ). When τ(u) = u −v is a shift, as we will verify in Section 4.2, the
operator ρ(τ) = Sv is a shift operator that commutes with convolution. Since
convolution operators are diagonalised by the Fourier transform, the action
of shift in the frequency domain amounts to shifting the complex phase of
the Fourier transform,
(d
Svx)(ξ) = e−iξvˆx(ξ).
Thus, the Fourier modulus f(x) = |ˆx| removing the complex phase is a simple
shift-invariant function, f(Svx) = f(x). However, if we have only approxi-
mate translation, τ(u) = u −˜τ(u) with ∥∇τ∥∞= supu∈Ω∥∇˜τ(u)∥≤ϵ, the
situation is entirely diﬀerent: it is possible to show that
∥f(ρ(τ)x) −f(x)∥
∥x∥
= O(1)
irrespective of how small ϵ is (i.e., how close is τ to being a shift). Conse-
quently, such Fourier representation is unstable under deformations, however
small. This unstability is manifested in general domains and non-rigid trans-
formations; we will see another instance of this unstability in the analysis
of 3d shapes using the natural extension of Fourier transforms described in
Section 4.4.
Wavelets oﬀer a remedy to this problem that also reveals the power of multi-
scale representations. In the above example, we can show (Mallat, 2012) that
the wavelet decomposition Wψx is approximately equivariant to deformations,
∥ρ(τ)(Wψx) −Wψ(ρ(τ)x)∥
∥x∥
= O(ϵ).
This notation implies that
ρ(τ) acts on the spatial
coordinate of (Wψx)(u, ξ).
In other words, decomposing the signal information into scales using lo-
calised ﬁlters rather than frequencies turns a global unstable representation
into a family of locally stable features. Importantly, such measurements at
diﬀerent scales are not yet invariant, and need to be progressively processed
towards the low frequencies, hinting at the deep compositional nature of
modern neural networks, and captured in our Blueprint for Geometric Deep
Learning, presented next.
Scale Separation Prior:
We can build from this insight by considering
a multiscale coarsening of the data domain Ωinto a hierarchy Ω1, . . . , ΩJ.
As it turns out, such coarsening can be deﬁned on very general domains,


--- Page 30 ---
26
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
Figure 7: Illustration of Scale Separation for image classiﬁcation tasks. The
classiﬁer f′ deﬁned on signals on the coarse grid X(Ω′) should satisfy f ≈
f′ ◦P, where P : X(Ω) →X(Ω′).
including grids, graphs, and manifolds. Informally, a coarsening assimilates
nearby points u, u′ ∈Ωtogether, and thus only requires an appropriate
notion of metric in the domain. If Xj(Ωj, Cj) := {xj : Ωj →Cj} denotes
signals deﬁned over the coarsened domain Ωj, we informally say that a
function f : X(Ω) →Y is locally stable at scale j if it admits a factorisation
of the form f ≈fj ◦Pj, where Pj : X(Ω) →Xj(Ωj) is a non-linear coarse
graining and fj : Xj(Ωj) →Y. In other words, while the target function f
might depend on complex long-range interactions between features over
the whole domain, in locally-stable functions it is possible to separate the
interactions across scales, by ﬁrst focusing on localised interactions that are
then propagated towards the coarse scales.
Such principles
Fast Multipole Method
(FMM) is a numerical
technique originally
developed to speed up the
calculation of long-ranged
forces in n-body problems.
FMM groups sources that lie
close together and treats
them as a single source.
are of fundamental importance in many areas of physics and
mathematics, as manifested for instance in statistical physics in the so-called
renormalisation group, or leveraged in important numerical algorithms such
as the Fast Multipole Method. In machine learning, multiscale represen-
tations and local invariance are the fundamental mathematical principles
underpinning the eﬃciency of Convolutional Neural Networks and Graph
Neural Networks and are typically implemented in the form of local pooling.
In future work, we will further develop tools from computational harmonic
analysis that unify these principles across our geometric domains and will
shed light onto the statistical learning beneﬁts of scale separation.


--- Page 31 ---
3. GEOMETRIC PRIORS
27
3.5
The Blueprint of Geometric Deep Learning
The geometric principles of Symmetry, Geometric Stability, and Scale Sepa-
ration discussed in Sections 3.1–3.4 can be combined to provide a universal
blueprint for learning stable representations of high-dimensional data. These
representations will be produced by functions f operating on signals X(Ω, C)
deﬁned on the domain Ω, which is endowed with a symmetry group G.
The geometric priors we have described so far do not prescribe a speciﬁc
architecture for building such representation, but rather a series of necessary
conditions. However, they hint at an axiomatic construction that provably
satisﬁes these geometric priors, while ensuring a highly expressive represen-
tation that can approximate any target function satisfying such priors.
A simple initial observation is that, in order to obtain a highly expressive
representation, we are required to introduce a non-linear element, since if f
is linear and G-invariant, then for all x ∈X(Ω),
Here, µ(g) is known as the
Haar measure of the group G,
and the integral is performed
over the entire group.
f(x) =
1
µ(G)
Z
G
f(g.x)dµ(g) = f

1
µ(G)
Z
G
(g.x)dµ(g)

,
which indicates that F only depends on x through the G-average Ax =
1
µ(G)
R
G(g.x)dµ(g). In the case of images and translation, this would entail
using only the average RGB color of the input!
While this reasoning shows that the family of linear invariants is not a very rich
object, the family of linear equivariants provides a much more powerful tool,
since it enables the construction of rich and stable features by composition
with appropriate non-linear maps, as we will now explain. Indeed, if B :
X(Ω, C) →X(Ω, C′) is G-equivariant satisfying B(g.x) = g.B(x) for all
x ∈X and g ∈G, and σ : C′ →C′′ is an arbitrary (non-linear) map, then
we easily verify that the composition U := (σ ◦B) : X(Ω, C) →X(Ω, C′′)
is also G-equivariant, where σ : X(Ω, C′) →X(Ω, C′′) is the element-wise
instantiation of σ given as (σ(x))(u) := σ(x(u)).
This simple property allows us to deﬁne a very general family of G-invariants,
by composing U with the group averages A ◦U : X(Ω, C) →C′′. A natural
question is thus whether any G-invariant function can be approximated at
arbitrary precision by such a model, for appropriate choices of B and σ. It
is not hard to adapt the standard Universal Approximation Theorems from
unstructured vector inputs to show that shallow ‘geometric’ networks are


--- Page 32 ---
28
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
Figure 8: Geometric Deep Learning blueprint, exempliﬁed on a graph. A typi-
cal Graph Neural Network architecture may contain permutation equivariant
layers (computing node-wise features), local pooling (graph coarsening),
and a permutation-invariant global pooling layer (readout layer).
also universal approximators, by properly generalising the group average
to a general non-linear invariant.
Such proofs have been
demonstrated, for example,
for the Deep Sets model by
Zaheer et al. (2017).
However, as already described in the
case of Fourier versus Wavelet invariants, there is a fundamental tension
between shallow global invariance and deformation stability. This motivates
an alternative representation, which considers instead localised equivariant
maps.
Meaningful metrics can be
deﬁned on grids, graphs,
manifolds, and groups. A
notable exception are sets,
where there is no predeﬁned
notion of metric.
Assuming that Ωis further equipped with a distance metric d, we
call an equivariant map U localised if (Ux)(u) depends only on the values
of x(v) for Nu = {v : d(u, v) ≤r}, for some small radius r; the latter set Nu
is called the receptive ﬁeld.
A single layer of local equivariant map U cannot approximate functions with
long-range interactions, but a composition of several local equivariant maps
UJ ◦UJ−1 · · · ◦U1 increases the receptive ﬁeld
The term ‘receptive ﬁeld’
originated in the
neuroscience literature,
referring to the spatial
domain that aﬀects the
output of a given neuron.
while preserving the stability
properties of local equivariants. The receptive ﬁeld is further increased
by interleaving downsampling operators that coarsen the domain (again
assuming a metric structure), completing the parallel with Multiresolution
Analysis (MRA, see e.g. Mallat (1999)).
In summary, the geometry of the input domain, with knowledge of an un-
deryling symmetry group, provides three key building blocks: (i) a local
equivariant map, (ii) a global invariant map, and (iii) a coarsening operator.


--- Page 33 ---
3. GEOMETRIC PRIORS
29
These building blocks provide a rich function approximation space with
prescribed invariance and stability properties by combining them together
in a scheme we refer to as the Geometric Deep Learning Blueprint (Figure 8).
Geometric Deep Learning Blueprint
Let Ωand Ω′ be domains, G a symmetry group over Ω, and write Ω′ ⊆Ω
if Ω′ can be considered a compact version of Ω.
We deﬁne the following building blocks:
Linear G-equivariant layer B
:
X(Ω, C)
→
X(Ω′, C′) satisfying
B(g.x) = g.B(x) for all g ∈G and x ∈X(Ω, C).
Nonlinearity σ : C →C′ applied element-wise as (σ(x))(u) = σ(x(u)).
Local pooling (coarsening) P : X(Ω, C) →X(Ω′, C), such that Ω′ ⊆Ω.
G-invariant layer (global pooling)
A
:
X(Ω, C)
→
Y satisfying
A(g.x) = A(x) for all g ∈G and x ∈X(Ω, C).
Using these blocks allows constructing G-invariant functions f
:
X(Ω, C) →Y of the form
f = A ◦σJ ◦BJ ◦PJ−1 ◦. . . ◦P1 ◦σ1 ◦B1
where the blocks are selected such that the output space of each block
matches the input space of the next one. Diﬀerent blocks may exploit
diﬀerent choices of symmetry groups G.
Diﬀerent settings of Geometric Deep Learning
One can make an impor-
tant distinction between the setting when the domain Ωis assumed to be ﬁxed
and one is only interested in varying input signals deﬁned on that domain,
or the domain is part of the input as varies together with signals deﬁned on
it. A classical instance of the former case is encountered in computer vision
applications, where images are assumed to be deﬁned on a ﬁxed domain
