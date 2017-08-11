# 1. Introduction
Representation of words as vectors in a high dimensional vector space is a vital
step of the state-of-the-art natural language processing techniques, which employ
advanced matrix computations to gain insight into semantic and syntactic similarities
between words and/or documents. The two main language models in this regard are 
*context-counting* and *context-predicting* models.

Context-counting models construct a word co-occurrence matrix from a context window
in a given corpus. The underlying assumption is that similar words will appear in the same context, and hence their vector representations will be closer to each other. However, the size of raw co-occurrence matrices grows exponentially with the vocabulary. For this and other performance reasons, dimensionality reduction techniques such as \emph{Latent Semantic Indexing} are often used in context-counting models together with \emph{Weighting} techniques \cite{Manning:1999:FSN:311445,ARIS:ARIS1440380105}.

On the other hand, context-predicting models approach the word embedding problem as a supervised learning task, which tries to predict the vector embedding of a target word directly from the context. That is, given the embedding of other words in the context it 
learns the embedding of the target word (or vise versa) (Bengio, 2003 and Mikolov, 2013). Baroni et al. conducted an extensive comparison between count and predictive models in (Marcobaroni, 2014)
and reported that the predictive models performed significantly better than the count models in a number of evaluation tasks.

Bengio et al. proposed one of the first predictive models, which is a *feedforward Neural Network Language Model (NNLM)* (Bengio, 2003). The end result 
of an NNLM is to learn a probability distribution for words given other words in a context.
Architecturally, it has four layers. The first layer is an input layer where **N** previous words of a context are encoded using one-hot encoding. Layer 2: Is a projection layer where each word in the vocabulary is represented in a high dimensional space. This is a dense representation of the input, where each of the **N** context words are represented by a **D**-dimensional vector. That is, this layer produces an **N × D** matrix.
Layer 3 is a hidden layer in a normal neural network. If the hidden layer has $H$ nodes, there will be **H × N** matrix of weights. Layer 4 is the output layer of the neural network. At this layer, probability is assigned to each word in the vocabulary.

More recently, Mikolov et al. proposed a simple but yet efficient neural network model (Mikolov, 2013), which is implemented as the [**Word2Vec toolkit**](https://code.google.com/archive/p/word2vec/). In contrast to NNLM,   Word2Vec does not use the hidden layer. For a clear and illustrative 
explanation of the Word2Vec toolkit see (Rong, 2014). In the following section, we will discuss the two main architectures and parameter settings in Word2Vec.

In this work, we explore the different parameter settings in Word2Vec and their effect in 
model training time and accuracy using a preprocessed Wikipedia (English) text corpus. Additionally, a [model](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)pre-trained on the Google news data set is also used.

# 2. Word2Vec architectures and parameters
There are two distinct architecture choices in Word2Vec: *Continuous Bag of Words (CBOW)* and *Skip-gram*. In a CBOW model, the objective is to predict the current word
given the context from words in the history. On the other hand, Skip-gram predicts the words within a context window given the current word. 

In addition to architecture, some parameter choices also impact the training time and/or accuracy of a Word2Vec model. The main parameters are discussed below.

**Training algorithm.** The training algorithm can be *hierarchical softmax*
or *negative sampling*. Hierarchical softmax attempts to maximize the log-likelihood of a target word, whereas negative sampling minimizes the log-likelihood of negative samples.

**Window size.** This is the context window size. That is, the maximum number of words to take into context when predicting the current word (CBOW) or the maximum number of predictions to make per word (Skip-gram).

**Down sampling.** A probability for randomly discarding frequent words
aimed to reduce the imbalance between most common and rare words.

**Size.** This is the number of feature vectors used to represent words, that is the dimensionality of vector space.

# 3. Methods
To evaluate the performance of the new and pre-trained word embeddings, three evaluation methods are applied using data provided in the course folder.

\noindent \textbf{\emph{The nearest neighbors evaluation task.}} This method is useful in testing the syntactic and semantic similarity of word vectors and their nearest neighbors using (cosine) distance
as a measure. However, the evaluation is limited to a subjective qualitative study. Although all the models are tested using all the words in this set, only a subset of the results are discussed in this report.

\noindent \textbf{\emph{The analogical reasoning evaluation task.}} This task set, originally from \cite{mikolov2013efficient}, tests the ability of a model to identify analogies such as 
\emph{If Paris is to France, Helsinki is to \underline{\textcolor{white}{aljdlfj}}}. Out of the available fourteen categories of analogical tests, the following five sets are
used in this project:
\begin{itemize}
	\item \emph{Capital-world.} Similar to the example given above, this set tests whether the model can correctly make an analogy between countries in the world and their capitals. 
	\item \emph{City-in-state.} This set contains analogies between
	states and their cities in the United States. 
	\item \emph{Currency.} This set evaluates a model on financial analogies like \emph{United States is to Dollar, Finland is to \underline{\textcolor{white}{aljdlfj}}}
	\item \emph{Family.}  This set evaluates a model on family analogies like \emph{boy is to girl, brother is to \underline{\textcolor{white}{aljdlfj}}}
	\item \emph{Opposite.} This set evaluates whether a model is able to correctly identify the opposite of a given word by analogy. The test phrases are similar to: \emph{clear is to unclear, efficient is to \underline{\textcolor{white}{aljdlfj}}}.
		
\end{itemize}

\noindent\textbf{\emph{The concrete noun categorization task.}} This evaluation set contains a list of $44$ concrete nouns and their categories in three levels. In the first level, each noun is categorized into two groups: \emph{"natural"} or \emph{"artifact."} The second level further categorizes the words into three groups: \emph{"animal","vegetable", \emph{and} "artifact"}. The last level is the most detailed of all, and categorizes the nouns into six groups: \emph{"bird","groundAnimal", "fruitTree", "green", "tool" \emph{and} "vehicle."}
For this task, the word vectors of all $44$ words are used in a K-means clustering algorithm\footnote{http://scikit-learn.org/stable/modules/clustering.html\#k-means} by setting the number of clusters to $2$, $3$, and $6$ corresponding to the three levels discussed above. The classification error is calculated by
\[
\text{error} = \frac{C(\text{misclassified words})}{C(\text{words})},
\] 
\noindent where the numerator is the count of misclassified words and the denominator is the total number of words (=$44$). Accuracy of the model is then $(1-error)\times 100\%$. Here the performance of K-means clustering also impacts the performance of the model under study. However, the goal of this project is to study the impact of 
parameter values in the performance of Word2Vec models. Thus, the assumption is that the effect of K-means on the performance of a model is uniform across different parameter settings.


# 4. Experiments
The main goal of this project is to evaluate the performance of Word2Vec models. To this end, two approaches are used. First, the performance of a pre-trained word embedding is evaluated. Second, we train a Word2Vec model by 
selecting a number of parameter values. For this part, we selected the following parameters and chose two values 
per each parameter:

\begin{itemize}
	\item \textbf{Architecture:} CBOW and Skip-gram
	\item \textbf{Window size:} $\{5, 10\}$
	\item \textbf{Down sampling:} $\{0.001, 0.00001\}$
	\item \textbf{Feature vectors:} $\{200, 400\}$
\end{itemize}

\noindent So, a total of $16$ models are trained and their performance, both in terms of training time and accuracy, is studied. The experiment is conducted on a computer with a 4 core Intel Xeon (R) E5345 CPU and 6 GB of RAM running on Ubuntu 16.04.

# 5. Results

\subsection*{5.1 Pre-trained model}
In this section, the performance of a pre-trained word embedding is discussed. The model was trained on part of Google News dataset, and contains 300-dimensional vectors for 3 million words and phrases. 
\begin{figure}
	\centering
	\includegraphics[trim=1cm 20cm 0cm 2cm, clip=true,width=\textwidth]{./Figures/google.pdf}
	\caption*{Table 1. Top 5 nearest words (labeled "First" to "Fifth") for the first ten words in the nearest neigbors evaluation set.}
	\label{fig:power}
	
\end{figure}

\begin{figure}
	\centering
	\includegraphics[trim=0.cm 0cm 0cm 0cm, clip=true,width=0.5\textwidth]{./Figures/language}
	\caption*{Figure 1. Top 15 nearest words to the word "language".}
	\label{fig:wordCloud}
\end{figure}	
	
\noindent\textbf{\emph{Nearest neighbors.}}  Table 1 shows the top five nearest words for the first ten words in the nearest neighbor evaluation set. The model was able to capture both syntactic (cat -- cats) and semantic (dog -- puppy) similarities between words. A word cloud visualization of the top 15 nearest words to the word "language" is depicted in Figure 1. Expected results such as "English" and "Arabic" are returned by the model. But the more interesting and unexpected word is "langauge". As the model was trained on a news data set, this result discloses the fact that "language" is often misspelled as "langauge". 

	\begin{figure}
		\includegraphics[trim=1.0cm 23cm 2cm 2cm, clip=true,width=0.9\textwidth]{./Figures/google-ana.pdf}
		\caption*{Table 2. Analogical reasoning task performance}
		\label{fig:analogical}
	\end{figure}

	\begin{figure}
		\centering
		\includegraphics[trim=2.0cm 25cm 2cm 2cm, clip=true,width=0.9\textwidth]{./Figures/google-cluster.pdf}
		\caption*{Table 3. Concrete noun classification task performance}
		\label{fig:class}
	\end{figure}

\noindent\textbf{\emph{Analogical reasoning task.}} This evaluation is based on the five analogical reasoning evaluation sets, namely: the \emph{Capital-world, City-in-state, Currency, Family, and Opposite} sets. Word2Vec comes with a built-in 

\noindent\emph{"accuracy"} function that runs such analogical evaluations and returns a JSON-like data structure of correct and incorrect analogies returned by the model.
From this, a single accuracy percentage can be reported using 
\[
\text{accuracy} = 100 \times \frac{\text{correct analogies}}{\text{Total analogies}},
\]
\noindent see \cite{accuracy} for more on this. These numbers are given in Table 2 for the five analogy task sets. The model performed poorly on the \emph{Currency} set, which maybe due to
the lack of financial topics in the training data.

\begin{figure*}
	\centering
	\begin{subfigure}[b]{0.49\textwidth}
		\includegraphics[width=0.9\textwidth]{./Figures/TrainTime-Down_sampling.pdf}
		\caption{}
		\label{fig:ws}
	\end{subfigure}
	\begin{subfigure}[b]{0.49\textwidth}
		\centering
		\includegraphics[width=0.9\textwidth]{./Figures/TrainTime-Window_size.pdf}
		\caption{}
		\label{fig:ds}
	\end{subfigure}\\
	\begin{subfigure}[b]{0.49\textwidth}
		\centering
		\includegraphics[width=0.9\textwidth]{./Figures/TrainTime-Feature_vectors.pdf}
		\caption{}
		\label{fig:fv}
	\end{subfigure}\\
	\caption*{Figure 2. Training time.}
	\label{fig:traintime}
\end{figure*}

\noindent\textbf{\emph{Concrete noun classification task.}} Here the task is to classify the set of given concrete nouns into two, three, and six clusters. Classification accuracy at each level is given in Table 3. Recall that the two clusters are representing "natural" and "artifact". Thus, the model was able to correctly classify  $97.7 \%$ of the concrete into "natural" and "artifact", whereas the classification into "animal", "vegetable", and "artifact" is correctly done for $95.5 \%$ of the words. 

\subsection*{5.2 Models trained on the Wikipedia English corpus}
As described in Section 4, two values are selected for each of the \emph{window size, down sampling, \emph{and} feature vectors (size)} parameters under both CBOW and Skip-gram. This results in 
$16$ different parameter combinations. Below, we discuss the performance of each of these.

\noindent\textbf{\emph{Training time.}} Figure 2 illustrates the training times pivoted around 
the architecture (CBOW or Skip-gram). The Skip-gram models (irrespective of the other parameter 
values) roughly took four times longer training time. When it comes to the parameters, window 
size (Figure 2b) and feature vectors (Figure 2c) appear to have larger impact on training time 
than down sampling(Figure 2a).

\noindent\textbf{\emph{Nearest neighbors.}} The large number of models makes it difficult to 
give a compact analysis of the nearest words data set. However, a random visual check was 
performed to verify the sanity of the trained models. CSV files containing the nearest words of all the words for all the models are available and can be shared upon request.
%Moreover, the top 5 nearest words of the first 40 words is given in Appendix A for the Skip-gram model with window size = 10, down sampling = 0.001 and feature vectors = 400.

\noindent\textbf{\emph{Analogical reasoning.}} The performance of all 16 models on the google analogical reasoning task is shown in Table 4. Looking at the \emph{currency} column, one can quickly see that all the models consistently gave very poor performance (an order of magnitude
poorer than the google embedding discussed in the previous section). Despite the significant 
amount of training time difference between the CBOW and Skip-gram models, they achieved equivalent performance in the analogical reasoning task, except for the \emph{capital-world} and \emph{city-in-state} sets, where the Skip-gram modes have a marginal edge over the CBOW.

\begin{figure}
	\centering
	\includegraphics[trim=1cm 13cm 0cm 3.25cm, clip=true,width=\textwidth]{./Figures/summary-analogical.pdf}
	\caption*{Table 4. Accuracy (in $\%$) of analogical reasoning task result.}
	\label{fig:ana}
	
\end{figure}

\begin{figure}
	\centering
	\includegraphics[trim=1cm 13cm 0cm 3.25cm, clip=true,width=\textwidth]{./Figures/summary-categorical.pdf}
	\caption*{Table 5. Concrete noun categorization task result.}
	\label{fig:cata}
	
\end{figure}
\noindent\textbf{\emph{Concrete noun categorization.}} Table 5 shows the performance of all the 
embeddings on three of the concrete noun categorization tasks. Once again, the CBOW and Skip-gram 
models gave comparable result. All the models gave impressive performance in classifying the nouns into two clusters, that is, "natural" and "artifact". The best of these models performed better than the google embedding on the two cluster task ($100\%$ accuracy), although the google embedding gave better performance on the six cluster task.

\section*{6. Conclusion}
In this project we conducted a comparative study of word embeddings trained using the Word2Vec
tool. In addition to training a number of models using the Wikipedia English corpus, we also  used publicly available embedding pre-trained using the Google news data set. Models are trained using both CBOW and Skip-gram. Different parameter values are also tested for window size, down sampling, and the size of feature vectors. The Skip-gram models roughly took four times longer training time than the CBOW models. However, the Skip-gram models did not perform any better than the CBOW models in the analogical reasoning and concrete noun categorization evaluation tasks.
%-------------------------------------------------------------------------------
% REFERENCES
%-------------------------------------------------------------------------------
\newpage
%\section*{References}
%\addcontentsline{toc}{section}{References}
\bibliographystyle{ieeetran}
\bibliography{reference.bib}
