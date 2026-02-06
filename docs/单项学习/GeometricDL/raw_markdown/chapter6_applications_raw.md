# 6 Problems and Applications

--- Page 106 ---
102
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
the input content, substantial work also learnt more explicit ways to direct
attention to the input. A powerful algorithmically grounded way of doing
so is the pointer network of Vinyals et al. (2015), which proposes a simple
modiﬁcation of recurrent attention to allow for pointing over elements of
variable-sized inputs. These ﬁndings have then been generalised to the set2set
architecture (Vinyals et al., 2016), which generalises seq2seq models to
unordered sets, supported by pointer network-backed LSTMs.
6
Problems and Applications
Invariances and symmetries arise all too commonly across data originating
in the real world. Hence, it should come as no surprise that some of the
most popular applications of machine learning in the 21st century have
come about as a direct byproduct of Geometric Deep Learning, perhaps
sometimes without fully realising this fact. We would like to provide readers
with an overview—by no means comprehensive—of inﬂuential works in
Geometric Deep Learning and exciting and promising new applications.
Our motivation is twofold: to demonstrate speciﬁc instances of scientiﬁc and
industrial problems where the ﬁve geometric domains commonly arise, and
to serve additional motivation for further study of Geometric Deep Learning
principles and architectures.
Chemistry and Drug Design
One of the most promising applications of
representation learning on graphs is in computational chemistry and drug
development.
Many drugs are not designed
but discovered, often
serendipitously. The historic
source of a number of drugs
from the plant kingdom is
reﬂected in their names: e.g.,
the acetylsalicylic acid,
commonly known as aspirin,
is contained in the bark of the
willow tree (Salix alba),
whose medicinal properties
are known since antiquity.
Traditional drugs are small molecules that are designed to
chemically attach (‘bind’) to some target molecule, typically a protein, in
order to activate or disrupt some disease-related chemical process. Unfor-
tunately, drug development is an extremely long and expensive process: at
the time of writing, bringing a new drug to the market typically takes more
than a decade and costs more than a billion dollars. One of the reasons is
the cost of testing where many drugs fail at diﬀerent stages – less than 5% of
candidates make it to the last stage (see e.g. Gaudelet et al. (2020)).
Since the space of chemically synthesisable molecules is very large (estimated
around 1060), the search for candidate molecules with the right combina-
tion of properties such as target binding aﬃnity, low toxicity, solubility, etc.
cannot be done experimentally, and virtual or in silico screening (i.e., the use


--- Page 107 ---
6. PROBLEMS AND APPLICATIONS
103
of computational techniques to identify promising molecules), is employed.
Machine learning techniques play an increasingly more prominent role in
this task. A prominent example of the use of Geometric Deep Learning
for virtual drug screening was recently shown by Stokes et al. (2020) us-
ing a graph neural network trained to predict whether or not candidate
molecules inhibit growth
Molecular graph of Halicin.
in the model bacterium Escherichia coli, they were
able to eﬀectively discover that Halicin, a molecule originally indicated for
treating diabetes, is a highly potent antibiotic, even against bacteria strains
with known antibiotic resistance. This discovery was widely covered in both
scientiﬁc and popular press.
Speaking more broadly, the application of graph neural networks to molecules
modeled as graphs has been a very active ﬁeld, with multiple specialised
architectures proposed recently that are inspired by physics and e.g. in-
corporate equivariance to rotations and translations (see e.g. Thomas et al.
(2018); Anderson et al. (2019); Fuchs et al. (2020); Satorras et al. (2021)).
Further, Bapst et al. (2020) have successfully demonstrated the utility of
GNNs for predictively modelling the dynamics of glass, in a manner that
outperformed the previously available physics-based models. Historically,
many works in computational chemistry were precursors of modern graph
neural network architectures sharing many common traits with them.
Drug Repositioning
While generating entirely novel drug candidates is a
potentially viable approach, a faster and cheaper avenue for developing new
therapies is drug repositioning, which seeks to evaluate already-approved
drugs (either alone or in combinations) for a novel purpose. This often
signiﬁcantly decreases the amount of clinical evaluation that is necessary
to release the drug to the market. At some level of abstraction, the action
of drugs on the body biochemistry and their interactions between each
other and other biomolecules can be modeled as a graph, giving rise to the
concept of ‘network medicine’ coined by the prominent network scientist
Albert-László Barabási and advocating the use of biological networks (such
as protein-protein interactions and metabolic pathways) to develop new
therapies (Barabási et al., 2011).
Geometric Deep Learning oﬀers a modern take on this class of approaches. A
prominent early example is the work of Zitnik et al. (2018), who used graph
neural networks to predict side eﬀects in a form of drug repositioning known
as combinatorial therapy or polypharmacy, formulated as edge prediction in


--- Page 108 ---
104
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
a drug-drug interaction graph. The novel coronavirus pandemic, which
is largely ongoing at the time of writing this text, has sparked a particular
interest in attempting to apply such approaches against COVID-19 (Gysi
et al., 2020). Finally, we should note that drug repositioning is not necessarily
limited to synthetic molecules: Veselkov et al. (2019) applied similar ap-
proaches to drug-like molecules contained in food (since, as we mentioned,
many plant-based foods contain biological analogues of compounds used in
oncological therapy). One of the authors of this text is involved in a collabo-
ration adding a creative twist to this research, by partnering with a molecular
chef that designs exciting recipes based on the ‘hyperfood’ ingredients rich
in such drug-like molecules.
Protein biology
Since we have already mentioned proteins as drug targets,
lets us spend a few more moments on this topic. Proteins are arguably
among the most important biomolecules that have myriads of functions
in our body, including protection against pathogens (antibodies), giving
structure to our skin (collagen), transporting oxygen to cells (haemoglobin),
catalysing chemical reactions (enzymes), and signaling (many hormones are
proteins). Chemically speaking, a protein is a biopolymer, or a chain of small
building blocks called aminoacids that under the inﬂuence of electrostatic
forces fold into a complex 3D structure. It is this structure that endows the
protein with its functions,
A common metaphor, dating
back to the chemistry Nobel
laureate Emil Fischer is the
Schlüssel-Schloss-Prinzip
(‘key-lock principle’, 1894):
two proteins often only
interact if they have
geometrically and chemically
complementary structures.
and hence it is crucial to the understanding of
how proteins work and what they do. Since proteins are common targets for
drug therapies, the pharmaceutical industry has a keen interest in this ﬁeld.
A typical hierarchy of problems in protein bioinformatics is going from
protein sequence (a 1D string over an alphabet of of 20 diﬀerent amino acids)
to 3D structure (a problem known as ‘protein folding’) to function (‘protein
function prediction’). Recent approaches such as DeepMind’s AlphaFold
by Senior et al. (2020) used contact graphs to represent the protein structure.
Gligorijevic et al. (2020) showed that applying graph neural networks on
such graphs allows to achieve better function prediction than using purely
sequence-based methods.
Gainza et al. (2020) developed
Oncologial target PD-L1
protein surface (heat map
indicated the predicted
binding site) and the
designed binder (shown as
ribbon diagram).
a Geometric Deep Learning pipeline called
MaSIF predicting interactions between proteins from their 3D structure.
MaSIF models the protein as a molecular surface discretised as a mesh, argu-
ing that this representation is advantageous when dealing with interactions
as it allows to abstract the internal fold structure. The architecture was based


--- Page 109 ---
6. PROBLEMS AND APPLICATIONS
105
on mesh convolutional neural network operating on pre-computed chem-
ical and geometric features in small local geodesic patches. The network
was trained using a few thousand co-crystal protein 3D structures from the
Protein Data Bank to address multiple tasks, including interface prediction,
ligand classiﬁcation, and docking, and allowed to do de novo (‘from scratch’)
design of proteins that could in principle act as biological immunotherapy
drug against cancer – such proteins are designed to inhibit protein-protein
interactions (PPI) between parts of the programmed cell death protein com-
plex (PD-1/PD-L1) and give the immune system the ability to attack the
tumor cells.
Recommender Systems and Social Networks
The ﬁrst popularised large-
scale applications of graph representation learning have occurred within so-
cial networks, primarily in the context of recommender systems. Recommenders
are tasked with deciding which content to serve to users, potentially depend-
ing on their previous history of interactions on the service. This is typically
realised through a link prediction objective: supervise the embeddings of
various nodes (pieces of content) such that they are kept close together if
they are deemed related (e.g. commonly viewed together). Then the proxim-
ity of two embeddings (e.g. their inner product) can be interpreted as the
probability that they are linked by an edge in the content graph, and hence
for any content queried by users, one approach could serve its k nearest
neighbours in the embedding space.
Among the pioneers of this methodology is the American image sharing and
social media company Pinterest: besides presenting one of the ﬁrst successful
deployments of GNNs in production, their method, PinSage
Pinterest had also presented
follow-up work, PinnerSage
(Pal et al., 2020), which
eﬀectively integrates
user-speciﬁc contextual
information into the
recommender.
, successfully
made graph representation learning scalable to graphs of millions of nodes
and billions of edges (Ying et al., 2018). Related applications, particularly
in the space of product recommendations, soon followed. Popular GNN-
backed recommenders that are currently deployed in production include
Alibaba’s Aligraph (Zhu et al., 2019) and Amazon’s P-Companion (Hao
et al., 2020). In this way, graph deep learning is inﬂuencing millions of
people on a daily level.
Within the context of content analysis on social networks, another note-
worthy eﬀort is Fabula AI, which is among the ﬁrst GNN-based startups
to be acquired (in 2019, by Twitter). The startup, founded by one of the
authors of the text and his team, developed novel technology for detecting


--- Page 110 ---
106
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
misinformation on social networks (Monti et al., 2019). Fabula’s solution
consists of modelling the spread of a particular news item by the network
of users who shared it. The users are connected if one of them re-shared
the information from the other, but also if they follow each other on the
social network. This graph is then fed into a graph neural network, which
classiﬁes the entire graph as either ‘true’ or ‘fake’ content – with labels based
on agreement between fact-checking bodies. Besides demonstrating strong
predictive power which stabilises quickly (often within a few hours of the
news spreading), analysing the embeddings of individual user nodes re-
vealed clear clustering of users who tend to share incorrect information,
exemplifying the well-known ‘echo chamber’ eﬀect.
Traﬃc forecasting
Transportation networks are another area
A road network (top) with its
corresponding graph
representation (bottom).
where Geo-
metric Deep Learning techniques are already making an actionable impact
over billions of users worldwide. For example, on road networks, we can
observe intersections as nodes, and road segments as edges connecting
them—these edges can then be featurised by the road length, current or
historical speeds along their segment, and the like.
One standard prediction problem in this space is predicting the estimated time
of arrival (ETA): for a given candidate route, providing the expected travel
time necessary to traverse it. Such a problem is essential in this space, not only
for user-facing traﬃc recommendation apps, but also for enterprises (such
as food delivery or ride-sharing services) that leverage these predictions
within their own operations.
Graph neural networks
Several of the metropolitan
areas where GNNs are
serving queries within
Google Maps, with indicated
relative improvements in
prediction quality (40+% in
cities like Sydney).
have shown immense promise in this space as well:
they can, for example, be used to directly predict the ETA for a relevant
subgraph of the road network (eﬀectively, a graph regression task). Such
an approach was successfully leveraged by DeepMind, yielding a GNN-
based ETA predictor which is now deployed in production at Google Maps
(Derrow-Pinion et al., 2021), serving ETA queries in several major metropoli-
tan areas worldwide. Similar returns have been observed by the Baidu Maps
team, where travel time predictions are currently served by the ConSTGAT
model, which is itself based on a spatio-temporal variant of the graph atten-
tion network model (Fang et al., 2020).


--- Page 111 ---
6. PROBLEMS AND APPLICATIONS
107
Object recognition
A principal benchmark for machine learning
One example input image,
the likes of which can be
found in ImageNet,
representing the “tabby cat”
class.
techniques
in computer vision is the ability to classify a central object within a provided
image. The ImageNet large scale visual recognition challenge (Russakovsky
et al., 2015, ILSVRC) was an annual object classiﬁcation challenge that pro-
pelled much of the early development in Geometric Deep Learning. Im-
ageNet requires models to classify realistic images scraped from the Web
into one of 1000 categories: such categories are at the same time diverse
(covering both animate and inanimate objects), and speciﬁc (with many
classes focused on distinguishing various cat and dog breeds). Hence, good
performance on ImageNet often implies a solid level of feature extraction
from general photographs, which formed a foundation for various transfer
learning setups from pre-trained ImageNet models.
The success of convolutional neural networks on ImageNet—particularly
the AlexNet model of Krizhevsky et al. (2012), which swept ILSVRC 2012
by a large margin—has in a large way spearheaded the adoption of deep
learning as a whole, both in academia and in industry. Since then, CNNs
have consistently ranked on top of the ILSVRC, spawning many popular
architectures such as VGG-16 (Simonyan and Zisserman, 2014)
Interestingly, the VGG-16
architecture has sixteen
convolutional layers and is
denoted as “very deep” by
the authors. Subsequent
developments quickly scaled
up such models to hundreds
or even thousands of layers.
, Inception
(Szegedy et al., 2015) and ResNets (He et al., 2016), which have successfully
surpassed human-level performance on this task. The design decisions and
regularisation techniques employed by these architectures (such as rectiﬁed
linear activations (Nair and Hinton, 2010), dropout (Srivastava et al., 2014),
skip connections (He et al., 2016) and batch normalisation (Ioﬀe and Szegedy,
2015)) form the backbone of many of the eﬀective CNN models in use today.
Concurrently with object classiﬁcation, signiﬁcant progress had been made
on object detection; that is, isolating all objects of interest within an image,
and tagging them with certain classes. Such a task is relevant in a variety of
downstream problems, from image captioning all the way to autonomous
vehicles. It necessitates a more ﬁne-grained approach, as the predictions need
to be localised; as such, often, translation equivariant models have proven
their worth in this domain. One impactful example in this space includes
the R-CNN family of models (Girshick et al., 2014; Girshick, 2015; Ren et al.,
2015; He et al., 2017) whereas, in the related ﬁeld of semantic segmentation,
the SegNet model of Badrinarayanan et al. (2017) proved inﬂuential, with
its encoder-decoder architecture relying on the VGG-16 backbone.


--- Page 112 ---
108
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
Game playing
Convolutional neural networks also play a prominent role
as translation-invariant feature extractors in reinforcement learning (RL) envi-
ronments, whenever the observed state can be represented in a grid domain;
e.g. this is the case when learning to play video games from pixels. In this
case, the CNN is responsible for reducing the input to a ﬂat vector represen-
tation, which is then used for deriving policy or value functions that drive the
RL agent’s behaviour. While the speciﬁcs of reinforcement learning are not
the focus of this section, we do note that some of the most impactful results
of deep learning in the past decade have come about through CNN-backed
reinforcement learning.
One particular example that is certainly worth mentioning here is Deep-
Mind’s AlphaGo (Silver et al., 2016). It encodes the current state within a
game of Go by applying a CNN to the 19 × 19 grid representing the current
positions of the placed stones. Then, through a combination of learning from
previous expert moves, Monte Carlo tree search, and self-play, it had suc-
cessfully reached a level of Go mastery that was suﬃcient to outperform Lee
Sedol, one of the strongest Go players of all time, in a ﬁve-round challenge
match that was widely publicised worldwide.
While this already represented a signiﬁcant milestone for broader artiﬁcial
intelligence—with Go having a substantially more complex state-space than,
say, chess
The game of Go is played on
a 19 × 19 board, with two
players placing white and
black stones on empty ﬁelds.
The number of legal states
has been estimated at
≈2 × 10170 (Tromp and
Farnebäck, 2006), vastly
outnumbering the number of
atoms in the universe.
—the development of AlphaGo did not stop there. The authors
gradually removed more and more Go-speciﬁc biases from the architecture,
with AlphaGo Zero removing human biases, optimising purely through self-
play (Silver et al., 2017), AlphaZero expands this algorithm to related two-
player games, such as Chess and Shogi; lastly, MuZero (Schrittwieser et al.,
2020) incorporates a model that enables learning the rules of the game on-
the-ﬂy, which allows reaching strong performance in the Atari 2600 console,
as well as Go, Chess and Shogi, without any upfront knowledge of the rules.
Throughout all of these developments, CNNs remained the backbone behind
these models’ representation of the input.
While several high-performing RL agents were proposed for the Atari 2600
platform over the years (Mnih et al., 2015, 2016; Schulman et al., 2017), for a
long time they were unable to reach human-level performance on all of the
57 games provided therein. This barrier was ﬁnally broken with Agent57
(Badia et al., 2020), which used a parametric family of policies, ranging
from strongly exploratory to purely exploitative, and prioritising them in
diﬀerent ways during diﬀerent stages of training. It, too, powers most of its


--- Page 113 ---
6. PROBLEMS AND APPLICATIONS
109
computations by a CNN applied to the video game’s framebuﬀer.
Text and speech synthesis
Besides images (which naturally map to a two-
dimensional grid), several of (geometric) deep learning’s strongest successes
have happened on one-dimensional grids. Natural examples of this are text
and speech, folding the Geometric Deep Learning blueprint within diverse
areas such as natural language processing and digital signal processing.
Some of the most widely applied and publicised works in this space focus
on synthesis: being able to generate speech or text, either unconditionally or
conditioned on a particular prompt. Such a setup can support a plethora of
useful tasks, such as text-to-speech (TTS), predictive text completion, and ma-
chine translation. Various neural architectures for text and speech generation
have been proposed over the past decade, initially mostly based on recurrent
neural networks (e.g. the aforementioned seq2seq model (Sutskever et al.,
2014) or recurrent attention (Bahdanau et al., 2014)). However, in recent
times, they have been gradually replaced by convolutional neural networks
and Transformer-based architectures.
One particular limitation of simple 1D convolutions in this setting is their
linearly growing receptive ﬁeld, requiring many layers in order to cover the se-
quence generated so far. Dilated
Dilated convolution is also
referred to as à trous
convolution (literally “holed”
in French).
convolutions, instead, oﬀer an exponentially
growing receptive ﬁeld with an equivalent number of parameters. Owing to
this, they proved a very strong alternative, eventually becoming competitive
with RNNs on machine translation (Kalchbrenner et al., 2016), while drasti-
cally reducing the computational complexity, owing to their parallelisability
across all input positions.
Such techniques have also
outperformed RNNs on
problems as diverse as
protein-protein interaction
(Deac et al., 2019).
The most well-known application of dilated con-
volutions is the WaveNet model from van den Oord et al. (2016a). WaveNets
demonstrated that, using dilations, it is possible to synthesise speech at the
level of raw waveform (typically 16,000 samples per second or more), produc-
ing speech samples that were signiﬁcantly more “human-like” than the best
previous text-to-speech (TTS) systems
Besides this, the WaveNet
model proved capable of
generating piano pieces.
. Subsequently, it was further demon-
strated that the computations of WaveNets can be distilled in a much simpler
model, the WaveRNN (Kalchbrenner et al., 2018)—and this model enabled
eﬀectively deploying this technology at an industrial scale. This allowed not
only its deployment for large-scale speech generation for services such as
the Google Assistant, but also allowing for eﬃcient on-device computations;
e.g. for Google Duo, which uses end-to-end encryption.


--- Page 114 ---
110
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
Transformers (Vaswani et al., 2017) have managed to surpass the limitations
of both recurrent and convolutional architectures, showing that self-attention
is suﬃcient for achieving state-of-the-art performance in machine transla-
tion. Subsequently, they have revolutionised natural language processing.
Through the pre-trained embeddings provided by models such as BERT (De-
vlin et al., 2018), Transformer computations have become enabled for a large
amount of downstream applications of natural language processing—for
example, Google uses BERT embeddings to power its search engine.
Arguably the most widely publicised application of Transformers in the
past years is text generation, spurred primarily by the Generative Pre-trained
Transformer (GPT, Radford et al. (2018, 2019); Brown et al. (2020)) family of
models from OpenAI. In particular, GPT-3 (Brown et al., 2020) successfully
scaled language model learning to 175 billion learnable parameters, trained
on next-word prediction on web-scale amounts of scraped textual corpora.
This allowed it not only to become a highly-potent few-shot learner on a
variety of language-based tasks, but also a text generator with capability to
produce coherent and human-sounding pieces of text. This capability not
only implied a large amount of downstream applications, but also induced
a vast media coverage.
Healthcare
Applications in the medical domain are another promising
area for Geometric Deep Learning. There are multiple ways in which these
methods are being used. First, more traditional architectures such as CNNs
have been applied to grid-structured data, for example, for the prediction of
length of stay in Intensive Care Units (Rocheteau et al., 2020), or diagnosis of
sight-threatening diseases from retinal scans (De Fauw et al., 2018). Winkels
and Cohen (2019) showed that using 3D roto-translation group convolutional
networks improves the accuracy of pulmonary nodule detection compared
to conventional CNNs.
Second, modelling organs as geometric surfaces, mesh convolutional neural
networks were shown to be able to address a diverse range of tasks, from
reconstructing facial structure from genetics-related information (Mahdi
et al., 2020) to brain cortex parcellation (Cucurull et al., 2018) to regressing
demographic properties from cortical surface structures (Besson et al., 2020).
The latter examples represent an increasing trend in neuroscience to consider
the brain as a surface with complex folds
Such structure of the brain
cortex are called sulci and
gyri in anatomical literature.
giving rise to highly non-Euclidean
structures.


--- Page 115 ---
6. PROBLEMS AND APPLICATIONS
111
At the same time, neuroscientists often try construct and analyse functional
networks of the brain representing the various regions of the brain that are
activated together when performing some cognitive function; these networks
are often constructed using functional magnetic resonance imaging (fMRI)
that shows in real time which areas of the brain consume more blood.
Typically, Blood
Oxygen-Level Dependent
(BOLD) contrast imaging is
used.
These
functional networks can reveal patient demographics (e.g., telling apart
males from females, Arslan et al. (2018)), as well as used for neuropathology
diagnosis, which is the third area of application of Geometric Deep Learning
in medicine we would like to highlight here. In this context, Ktena et al.
(2017) pioneered the use of graph neural networks for the prediction of
neurological conditions such as Autism Spectrum Disorder. The geometric
and functional structure of the brain appears to be intimately related, and
recently Itani and Thanou (2021) pointed to the beneﬁts of exploiting them
jointly in neurological disease analysis.
Fourth, patient networks are becoming more prominent in ML-based medical
diagnosis. The rationale behind these methods is that the information of
patient demographic, genotypic, and phenotypic similarity could improve
predicting their disease. Parisot et al. (2018) applied graph neural networks
on networks of patients created from demographic features for neurological
disease diagnosis, showing that the use of the graph improves prediction
results. Cosmo et al. (2020) showed the beneﬁts of latent graph learning
(by which the network learns an unknown patient graph) in this setting.
The latter work used data from the UK Biobank, a large-scale collection of
medical data including brain imaging (Miller et al., 2016).
A wealth of data about hospital patients may be found in electronic health
records (EHRs)
Publicly available
anonymised critical-care EHR
datasets include MIMIC-III
(Johnson et al., 2016) and
eICU (Pollard et al., 2018).
. Besides giving a comprehensive view of the patient’s pro-
gression, EHR analysis allows for relating similar patients together. This
aligns with the pattern recognition method, which is commonly used in diag-
nostics. Therein, the clinician uses experience to recognise a pattern of clinical
characteristics, and it may be the primary method used when the clinician’s
experience may enable them to diagnose the condition quickly. Along these
lines, several works attempt to construct a patient graph based on EHR data,
either by analysing the embeddings of their doctor’s notes (Malone et al.,
2018), diagnosis similarity on admission (Rocheteau et al., 2021), or even
assuming a fully-connected graph (Zhu and Razavian, 2019). In all cases,
promising results have been shown in favour of using graph representation
learning for processing EHRs.


--- Page 116 ---
112
BRONSTEIN, BRUNA, COHEN & VELIČKOVIĆ
Particle physics and astrophysics
High energy physicists were perhaps
among the ﬁrst domain experts in the ﬁeld of natural sciences to embrace
the new shiny tool, graph neural networks. In a recent review paper, Shlomi
et al. (2020)
Part of the Large Hadron
Collider detectors.
note that machine learning has historically been heavily used in
particle physics experiments, either to learn complicated inverse functions
allowing to infer the underlying physics process from the information mea-
sured in the detector, or to perform classiﬁcation and regression tasks. For
the latter, it was often necessary to force the data into an unnatural repre-
sentation such as grid, in order to be able to used standard deep learning
architectures such as CNN. Yet, many problems in physics involve data in
the form of unordered sets with rich relations and interactions, which can
be naturally represented as graphs.
One important application in high-energy physics is the reconstruction and
classiﬁcation of particle jets – sprays of stable particles arising from multiple
successive interaction and decays of particles originating from a single initial
event. In the Large Hardon Collider, the largest and best-known particle
accelerator built at CERN, such jet are the result of collisions of protons at
nearly the speed of light. These collisions produce massive particles, such as
the long though-for Higgs boson or the top quark. The identiﬁcation and
classiﬁcation of collision events is of crucial importance, as it might provide
experimental evidence to the existence of new particles.
Multiple Geometric Deep Learning approaches
Example of a particle jet.
have recently been proposed
for particle jet classiﬁcation task, e.g. by Komiske et al. (2019) and Qu and
Gouskos (2019), based on DeepSet and Dynamic Graph CNN architectures,
respectively. More recently, there has also been interest in developing spe-
cialsed architectures derived from physics consideration and incorporating
inductive biases consistent with Hamiltonian or Lagrangian mechanics (see
e.g. Sanchez-Gonzalez et al. (2019); Cranmer et al. (2020)), equivariant to
the Lorentz group (a fundamental symmetry of space and time in physics)
(Bogatskiy et al., 2020), or even incorporating symbolic reasoning (Cran-
mer et al., 2019) and capable of learning physical laws from data. Such
approaches are more interpretable (and thus considered more ‘trustworthy’
by domain experts) and also oﬀer better generalisation.
Besides particle accelerators, particle detectors are now being used by as-
trophysicist for multi-messenger astronomy – a new way of coordinated obser-
vation of disparate signals, such as electromagnetic radiation, gravitational
waves, and neutrinos, coming from the same source. Neutrino astronomy is


--- Page 117 ---
6. PROBLEMS AND APPLICATIONS
113
of particular interest, since neutrinos interact only very rarely with matter,
and thus travel enormous distances practically unaﬀected.
The characteristic pattern of
light deposition in IceCube
detector from background
events (muon bundles, left)
and astrophysical neutrinos
(high-energy single muon,
right). Choma et al. (2018)
Detecting neutri-
nos allows to observe objects inaccessible to optical telescopes, but requires
enormously-sized detectors – the IceCube neutrino observatory uses a cubic
kilometer of Antarctic ice shelf on the South Pole as its detector. Detecting
high-energy neutrinos can possibly shed lights on some of the most mysteri-
ous objects in the Universe, such as blazars and black holes. Choma et al.
(2018) used a Geometric neural network to model the irregular geometry of
the IceCube neutrino detector, showing signiﬁcantly better performance in
detecting neutrinos coming from astrophysical sources and separating them
from background events.
While neutrino astronomy oﬀers a big promise in the study of the Cosmos,
traditional optical and radio telescopes are still the ‘battle horses’ of as-
tronomers. With these traditional instruments, Geometric Deep Learning
can still oﬀer new methodologies for data analysis. For example, Scaife
and Porter (2021) used rotationally-equivariant CNNs for the classiﬁcation
of radio galaxies, and McEwen et al. (2021) used spherical CNNs for the
analysis of cosmic microwave background radiation, a relic from the Big
Bang that might shed light on the formation of the primordial Universe. As
we already mentioned, such signals are naturally represented on the sphere
and equivariant neural networks are an appropriate tool to study them.
Virtual and Augmented Reality
Another ﬁeld of applications which served
as the motivation for the development of a large class of Geometric Deep
Learning methods is computer vision and graphics, in particular, dealing
with 3D body models for virtual and augmented reality. Motion capture tech-
nology used to produce special eﬀects in movies like Avatar often operates
in two stages: ﬁrst, the input from a 3D scanner capturing the motions of the
body or the face of the actor is put into correspondence with some canonical
shape, typically modelled as a discrete manifold or a mesh (this problem is
often called ‘analysis’). Second, a new shape is generated to repeat the mo-
tion of the input (‘synthesis’). Initial works on Geometric Deep Learning in
computer graphics and vision (Masci et al., 2015; Boscaini et al., 2016a; Monti
et al., 2017) developed mesh convolutional neural networks to address the
analysis problem, or more speciﬁcally, deformable shape correspondence.
First geometric autoencoder architectures for 3D shape synthesis were pro-
posed independently by Litany et al. (2018) and Ranjan et al. (2018). In
