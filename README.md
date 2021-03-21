# Local search engine

Google nlp task @HackNU/2021. Done through following [this tutorial](https://www.kaggle.com/amitkumarjaiswal/nlp-search-engine) by [Amit Kumar Jaiswal](https://www.kaggle.com/amitkumarjaiswal) .

## Screenshot

![screenshot](https://user-images.githubusercontent.com/36531464/111894768-b2a62780-8a37-11eb-931a-c86a458faa2e.png)

## Getting started

1. Clone the repo:

```sh
   git clone https://github.com/dabarov/local-search-engine.git
```

- The provided dataset is necessary for scripts to run.

2. Download and extract dataset using [this link](https://drive.google.com/drive/folders/1NngVRs5IORpDLMj7bRPEXJ558u3rrVfO?usp=sharing)

- Your directory should look the following way:

```sh
.
├── kaggle-news-dataset
│   ├── articles1.csv
│   ├── articles2.csv
│   └── articles3.csv
├── LICENSE
├── main.py
├── preprocessor.py
├── README.md
└── worddic_1000.npy
```

3. Install Python3 packages:

```sh
nltk                          3.5
numpy                         1.18.1
pandas                        0.24.2
```

## Usage

- Execute the following command:

```sh
python3 main.py
```

- Wait several seconds for code to process word dictionary
- Type your request when you see the follwing line:

```sh
Search:
```

- It will give you the output with max 5 sources (Example with "presidential election" response):

```sh
Search: presidential election 


1. Countering Trump, Bipartisan Voices Strongly Affirm Findings on Russian Hacking - The New York Times
------------------
Authors: Matt Flegenheimer and Scott Shane
WASHINGTON  —   A united front of top intelligence officials and senators from both parties on Thurs ...


2. The Perfect Weapon: How Russian Cyberpower Invaded the U.S. - The New York Times
------------------
Authors: Eric Lipton, David E. Sanger and Scott Shane
WASHINGTON  —   When Special Agent Adrian Hawkins of the Federal Bureau of Investigation called the  ...


3. Illegal Voting Claims, and Why They Don’t Hold Up - The New York Times
------------------
Authors: Nate Cohn
There isn’t any evidence to support President Trump’s assertion that three to five million illegal v ...


4. China Assails U.S. Pledge to Defend Disputed Islands Controlled by Japan - The New York Times
------------------
Authors: Jane Perlez
BEIJING  —   China reacted with strong displeasure on Saturday to a promise by Defense Secretary Jim ...


5. Democrats, With Garland on Mind, Mobilize for Supreme Court Fight - The New York Times
------------------
Authors: Carl Hulse
WASHINGTON  —   Senate Democrats have one particular judge’s name in mind as they await the identity ...

```
