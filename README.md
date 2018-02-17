## Planning

1. Training Corpora Definition - 16 Feb
2. Corpora pre-processing - 28 Feb
3. Sentence tokenization - 12 Mar
4. Sentence vetorization - 26 Mar
5. Word to sentence database - 9 Apr
6. SOM of sentences - 23 Apr
7. Word fingerprint - 4 May
8. Text fingerprint - 17 May
9. Evaluation


## 1. Training Corpora Definition

Wikipedia dumps from [wikimedia 2018-01-01](https://dumps.wikimedia.org/enwiki/20180101/) 

### 1.1 Extracting plain text from Wikipedia dumps

[github - attardi/wikiextractor](https://github.com/attardi/wikiextractor)

Document files contains a series of Wikipedia articles, represented each by an XML doc element:
```markdown
<doc>...</doc>
<doc>...</doc>
...
<doc>...</doc>
```
The element doc has the following attributes:

- id, which identifies the document by means of a unique serial number
- url, which provides the URL of the original Wikipedia page.
The content of a doc element consists of pure text, one paragraph per line.

Example:
```markdown
<doc id="2" url="http://it.wikipedia.org/wiki/Harmonium">
Harmonium.
L'harmonium è uno strumento musicale azionato con una tastiera, detta manuale.
Sono stati costruiti anche alcuni harmonium con due manuali.
...
</doc>
```


## 9. Evaluation

Evaluation code repository: [github - kudkudak/word-embeddings-benchmarks](https://github.com/kudkudak/word-embeddings-benchmarks.git)
Evaluation methods: [arxiv.org/abs/1702.02170](https://arxiv.org/abs/1702.02170)

other alternative methods: [github - mfaruqui/eval-word-vectors](https://github.com/mfaruqui/eval-word-vectors)















### Markdown help

You can use the [editor on GitHub](https://github.com/avsilva/sparse-nlp/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.


Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

**Bold** and _Italic_ and `Code` text

![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/avsilva/sparse-nlp/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
