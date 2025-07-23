![](lectures/img/575_banner.png)

## Important links 

- [Course Jupyter book](https://pages.github.ubc.ca/mds-2024-25/DSCI_575_adv-mach-learn_students/README.html)
- [Course GitHub page](https://github.ubc.ca/MDS-2024-25/DSCI_575_adv-mach-learn_students)
- [Slack Channel](https://ubc-mds.slack.com/messages/575_adv-mach-learn)
- [Canvas](https://canvas.ubc.ca/courses/154208)
- [Gradescope](https://www.gradescope.ca/courses/26525)
- [Class + office hours calendar](https://ubc-mds.github.io/calendar/)

## Course learning outcomes    
In this course, we will learn some advanced machine learning methods in the context of natural language processing (NLP) applications, including Markov chains, hidden Markov models, recurrent neural networks, and self-attention and transformers. 
<details>
  <summary>Click to expand!</summary>     
    
By the end of the course, students will be able to:

- Perform basic text preprocessing.
- Define Markov chains and apply them for generation and inference.
- Explain the concept of stationary distribution in Markov chains.
- Describe Hidden Markov Models (HMMs) and perform inference and decoding.
- Summarize Recurrent Neural Networks (RNNs) and their variations.
- Explain self-attention and transformers and apply them in NLP tasks.
</details>

## Deliverables

<details>
  <summary>Click to expand!</summary>
    
The following deliverables will determine your course grade:

| Assessment       | Weight  | Where to submit|
| :---:            | :---:   |:---:  | 
| Lab Assignment 1 | 12%     | [Gradescope](https://www.gradescope.ca/courses/26525) |
| Lab Assignment 2 | 12%     | [Gradescope](https://www.gradescope.ca/courses/26525) |
| Lab Assignment 3 | 12%     | [Gradescope](https://www.gradescope.ca/courses/26525) |
| Lab Assignment 4 | 12%     | [Gradescope](https://www.gradescope.ca/courses/26525) |
| Class participation  |  2%     | iClicker Cloud |
| Quiz 1           | 25%     | [Canvas](https://canvas.ubc.ca/courses/154208)     |
| Quiz 2           | 25%     | [Canvas](https://canvas.ubc.ca/courses/154208)     |

See [Calendar](https://ubc-mds.github.io/calendar/) for the due dates. 
</details>

## Teaching team
<details>
  <summary>Click to expand!</summary>

| Role             |  Name  | 
| :--------------: | :--------------: |
| Lecture instructor Section 001 | Varada Kolhatkar |
| Lab Instructor     Section 001 | Varada Kolhatkar |
| Lecture instructor Section 002 | Varada Kolhatkar |
| Lab Instructor     Section 001 | Varada Kolhatkar |
| Teaching assistant | Afsoon Gharib Mombeini | 
| Teaching assistant | Matin Daghyani |
| Teaching assistant | Meltem Omur |
| Teaching assistant | Nima Hashemi |
| Teaching assistant | Tony L Fong	 |

</details>  


## Lecture schedule

This course will be run in person. We will meet three times every week: twice for lectures and once for the lab. You can refer to the [Calendar](https://ubc-mds.github.io/calendar/) for lecture and lab times and locations. Lectures of this course will be a combination traditional live lecturing, class activities, and pre-recorded videos. Drafts of the lecture notes for each week will be made available earlier in the week.  

This course occurs during **Block 6** in the 2024/25 school year.

| Lecture  | Topic  | Assigned videos/Readings | Resources and optional readings |
|-------|------------|-----------|-----------|
|   0   | [Course Information](lectures/notes/00_course-information.ipynb) | 📹  <li> Videos: [16.1](https://youtu.be/GTC_iLPCjdY) | |
|   1   | [Markov Models](lectures/notes/01_Markov-models.ipynb) | | <li> [Markov chains in action](http://setosa.io/ev/markov-chains/) </li> | 
|   2   | [Language models, PageRank, text preprocessing](lectures/notes/02_LMs-text-preprocessing.ipynb) | 📹  <li> Videos: [16.2](https://youtu.be/7W5Q8gzNPBc) | <li>[OpenAI GPT3 demo](https://www.youtube.com/embed/fZSFNUT6iY8)</li><li> [Dan Jurafsky's videos on PageRank](https://www.youtube.com/playlist?list=PLaZQkZp6WhWzSy3WKExE7656jBxfXJh3I)</li> <li>[Dan Jurafsky's video on tokenization](https://www.youtube.com/watch?v=pEwBjcYdcKw)</li>|
|   3  | [Hidden Markov models](lectures/notes/03_HMMs-intro.ipynb) |  |<li>[Nando de Freitas' lecture on HMMs](https://www.youtube.com/watch?v=jY2E6ExLxaw)</li> <li>[A gentle intro to HMMs by Eric Fosler-Lussier](http://di.ubi.pt/~jpaulo/competence/tutorials/hmm-tutorial-1.pdf)</li>|
|   4  | [HMMs decoding and inference](lectures/notes/04_Viterbi-Baum-Welch.ipynb)) | (optional) [HMM Baum-Welch](https://youtu.be/_m5KuZGOOVI) (unlisted) | <li>[Nando de Freitas' lecture on HMMs](https://www.youtube.com/watch?v=jY2E6ExLxaw)</li> <li>[A gentle intro to HMMs by Eric Fosler-Lussier](http://di.ubi.pt/~jpaulo/competence/tutorials/hmm-tutorial-1.pdf)</li>|
|   5   | [Introduction to Recurrent Neural Networks (RNNs)](lectures/notes/05_intro-to-RNNs.ipynb) |  | <li>[The Unreasonable Effectiveness of Recurrent Neural  Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)</li><li>Highly recommended: [Sequence Processing with Recurrent Networks](https://web.stanford.edu/~jurafsky/slp3/9.pdf)</li>|  
|   6   | [Introduction to Transformers](lectures/notes/06_intro-to-transformers.ipynb) |   📹  <li> Videos: [Introduction to Self-Attention](https://youtu.be/NJ_kTPwcaJU) | |
|   7   | [Applications of Transformers](lectures/notes/07_transformers-applications.ipynb) |  | |
|   8   | [Large Language Models](lectures/notes/08_LLMs.ipynb) |  | |


The labs are going to be in person. There will be a lot of opportunity for discussion and getting help during lab sessions. Please make good use of this time.  

## Installation

We are providing you with a `conda` environment file which is available [here](env-dsci-575.yml). You can download this file and create a conda environment for the course and activate it as follows. 

```
conda env create -f env-dsci-575.yml
conda activate 575
```

We've only attempted to install this environment file on a few machines, and you may encounter issues with certain packages from the `yml` file when executing the commands above. This is not uncommon and may suggest that the specified package version is not yet available for your operating system via `conda`. When this occurs, you have a couple of options:

1. Modify the local version of the `yml` file to remove the line containing that package.
2. Create the environment without that package. 
3. Activate the environment and install the package manually either with `conda install` or `pip install` in the environment.   

_Note that this is not a complete list of the packages we'll be using in the course and there might be a few packages you will be installing using `conda install` later in the course. But this is a good enough list to get you started._ 


## Course communication
<details>
  <summary>Click to expand!</summary>

We all are here to help you learn and succeed in the course and the program. Here is how we'll be communicating with each other during the course. 

### Clarifications on the lecture notes or lab questions

If there is any clarification on the lecture material or lab questions, I'll post a message on our course channel and tag you. **It is your responsibility to read the messages whenever you are tagged.** (I know that there are too many things for you to keep track of. You do not have to read all the messages but please make sure to carefully read the messages whenever you are tagged.) 

### Questions on lecture material or labs

If you have questions about the lecture material or lab questions please post them on the course Slack channel rather than direct messaging me or the TAs. Here are the advantages of doing so: 
- You'll get a quicker response. 
- Your classmates will benefit from the discussion. 

When you ask your question on the course channel, please avoid tagging the instructor unless it's specific for the instructor (e.g., if you notice some mistake in the lecture notes). If you tag a specific person, other teaching team members or your colleagues are discouraged to respond. This will decrease the response rate on the channel. 

Please use some consistent convention when you ask questions on Slack to facilitate easy search for others or future you. For example, if you want to ask a question on Exercise 3.2 from Lab 1, start your post with the label `lab1-ex2.3`. Or if you have a question on lecture 2 material, start your post with the label `lecture2`. Once the question is answered/solved, you can add "(solved)" tag before the label (e.g., (solved) `lab1-ex2.3`). Do not delete your post even if you figure out the answer on your own. The question and the discussion can still be beneficial to others.  

### Questions related to grading

For each deliverable, after I return grades, I'll let you know who has graded what in our course Slack by opening an issue in the course GitHub repository. If you have questions related to grading
- First, make sure your concerns are reasonable (read the ["Reasonable grading concerns" policy](https://ubc-mds.github.io/policies/)). 
- If you believe that your request is reasonable, open a regrade request on Gradescope. 
- If you are unable to resolve the issue with the TA, send a Slack message to the instructor, including the appropriate TA in the conversation. 

### Questions related to your personal situation or talking about sensitive information
 
I am open for a conversation with you. If you want to talk about anything sensitive, please direct message me on Slack (and tag me) rather than posting it on the course channel. It might take a while for me to get back to you, but I'll try my best to respond as soon as possible. 

</details>

## Reference Material
<details>
    <summary>Click to expand!</summary>   

### Online resources     
* [Google NLP API](https://cloud.google.com/natural-language/)
* [Stanford CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/syllabus.html)
* [LDA2vec: Word Embeddings in Topic Models](https://www.datacamp.com/community/tutorials/lda2vec-topic-model)
* [7 Types of Artificial Neural Networks for Natural Language Processing](https://www.kdnuggets.com/2017/10/7-types-artificial-neural-networks-natural-language-processing.html)
* https://distill.pub/
* [Model-Based Machine Learning](http://mbmlbook.com/toc.html)
* [RNNs in TensorFlow, a practical guide and undocumented features](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/)
* [A list of readings about RNNs](https://github.com/tensorflow/magenta/tree/master/magenta/reviews)
* For NLP in R, see [Julia Silge's blog](https://juliasilge.com/blog/) posts on sentiment analysis of Jane Austen novels: [part 1](https://juliasilge.com/blog/you-must-allow-me/), [part 2](https://juliasilge.com/blog/if-i-loved-nlp-less/), [part 3](https://juliasilge.com/blog/life-changing-magic/), [part 4](https://juliasilge.com/blog/term-frequency-tf-idf/).
* [RNN resources](https://github.com/ajhalthor/awesome-rnn)

### Books
* Jurafsky, D., & Martin, J. H. [Speech and language processing](https://web.stanford.edu/~jurafsky/slp3/).
* Goodfellow, I., Bengio, Y., Courville, A., & Bengio, Y. (2016). [Deep learning (Vol. 1)](http://www.deeplearningbook.org/). Cambridge: MIT press. 
* [Jacob Eisenstein. Natural Language Processing](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)
* Goldberg, Y. (2017). Neural network methods for natural language processing. Synthesis Lectures on Human Language Technologies, 10(1), 1-309. 
* Bird, S., Klein, E., & Loper, E. (2009). [Natural language processing with Python](http://www.nltk.org/book/). O'Reilly Media, Inc.

</details> 

## Policies

Please see the general [MDS policies](https://ubc-mds.github.io/policies/).
