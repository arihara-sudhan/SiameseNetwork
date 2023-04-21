<h1>SIAMESE NETWORK</h1>
<h3>ARIHARASUDHAN</h3>
<img src='https://assets.lybrate.com/q_auto:eco,f_auto,w_1200,h_720,c_fill,g_auto/imgs/product/health-wiki/bpages/Benefits-Of-Cherry.jpg' alt=''>
<h1>INTRODUCTION</h1>
<p>We may want to see whether two images (Faces) are similar or not. Siamese Neural Networks help us here. Using Siamese Network, we can distinguise between faces or any other objects. We know the face-matching-locking system in our phone. What would be the background operation ? (Talkin’ of the training process) You don’t use many photos to perform such training. You use a single or a less number or images for such a training.
We may need a lot of images for such a problem. Isn’t it fascinating when there is no need for that while we use Siamese Network for image verification ? And, it is not even a workload to train a bunch of images since just one person is going to access the phone and training on some images of him is not a sin. But, imagine when an organization uses such an authentication mechanism for it’s employees. The situation becomes a bit worse.
We need n number of images of each employees of the organization to train the network. And another important problem here is the so-called Scalability. If a new employee joins there, we need to retrain the classifier and the situation is HELL. A Simple but efficient solution here is using the SIMILARITY MODEL rather than a CLASSIFIER MODEL. This is where the Siamese Network comes to take a part in our journey.
Let’s make illustrations for a better understanding.<p>

<h1>THE IDEA BEHIND</h1>
<p>Now, we just want to know what to do.
Looking at the image above, we can observe that we pass two input images to CNNs of same configurations which have a convolutional layer, Max-Pooling and an FC Layer. Eventually,  feature vectors / embeddings are obtained from these networks. These vectors are to be compared in order to check for similarity. Those feature vectors are passed into a softmax classifier. These two vectors if they are of the same person should be pretty much similar.
On the other hand, if they are of different people, these two vectors should be different. The siamese network uses identical networks.
The difference is found between two vectors using the given formula. It will be small on having similar images and vice versa.</p>

<h1>TRIPLET LOSS</h1>
Computationally speaking, in all sort of neural networks, we define a loss function which must be minimized for a better training. Here, we define the so-called triplet loss. But, why the name “Triplet Loss”? We are about to deal with collection of three data on each epoch. What are those three ? They are Anchor, Positive and a Negative Image. Anchor is the one given to be verified or a ground truth. 
Positive is the one which is similar to the Anchor. Negative is the one which is dissimilar to the Anchor. So, what we can do to go on ? We need the difference between the f(A) and f(P) to be less and that of the f(A) and f(N) to be a bit high. Collectively, d(A,P) should be less than or equal to d(A,N).
||f(A)-f(P)||2 <= ||f(A)-f(N)||2
d(A,P) <= d(A,N)
If we get zero or any other  things, we might end up with equality in both the sides. To avoid this, a margin is used.
d(A,P) <= d(A,N) – α
d(A,P)-d(A,N)+α <= 0
Let D = d(A,P)-d(A,N)+α
Now, we can define the triplet loss as the following :
L(A,P,N) = max(0,D)
For n training samples, we can define the cost function as,
J = Σ(i=0 To m)max(0,D)
Now, let’s learn about contrastive loss which can also be used for this.



<h1>CONTRASTIVE LOSS</h1>
This is one of the first training objectives that was used for contrastive learning. It takes as input a pair of samples that are either similar or dissimilar, and it brings similar samples closer and dissimilar samples far apart. More formally, we suppose that we have a pair  and a label  that is equal to 0 if the samples are similar and 1 otherwise. To extract a low-dimensional representation of each sample, we use a Convolutional Neural Network  that encodes the input images  and  into an embedding space where  and . The contrastive loss is defined as:
Let’s implement the Siamese Network using PyTorch.
