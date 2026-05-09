# Endsem Regex Generation Pipeline

1. take raw natural-language / regex pairs,
2. clean and split them into train / validation / test sets,
3. build vocabularies from the training data,
4. convert examples into numeric sequences,
5. prepare everything for model training.

This README explains the workflow in plain language so the project can be understood from start to the training stage.

## What is in this folder?

- `datasets/` contains the raw data sources and the processed split files.
- `preprocess_data.py` merges the datasets, cleans the text, and writes the split files.
- `indexer_local.py` stores the vocabulary helper used to map text to numbers.
- `indexers.pkl` stores the saved vocabularies.

## What problem is this project solving?

The project tries to learn a mapping from a natural-language description to a regular expression.

For example, a sentence such as "lines that contain blue" should be turned into a regex that matches the same meaning. The model does not read raw text directly. It first needs the data converted into a consistent numeric format.

That is why preprocessing is such an important part of the project.

## Data sources

The raw data comes from three dataset folders:

- `KB13`
- `NL-RX-Synth`
- `NL-RX-Turk`

Each folder contains pairs of lines:

- `src.txt` for the natural-language description
- `targ.txt` for the matching regular expression

Each line in `src.txt` must stay aligned with the same line in `targ.txt`.
That means line 1 in `src.txt` matches line 1 in `targ.txt`, line 2 matches line 2, and so on.

## How the split files are made

The preprocessing script combines the three datasets into one pool of examples and then creates three splits:

- training set
- validation set
- test set

The process is done in a repeatable way by using a fixed random seed.

### Step 1: read the pairs

The script reads every source line and its matching target line as one pair.
The pair is treated as a single example so the natural-language description and the regex never get separated.

### Step 2: clean the text

Before splitting, the script removes extra newline characters and applies the project’s placeholder normalisation and token cleaning rules.

This makes the data more consistent. For example, repeated placeholder styles or stray spacing are turned into a standard form so the model sees fewer unnecessary variations.

### Step 3: shuffle the pairs

The script shuffles the full list of pairs using a fixed seed.

This is important because it mixes examples from all sources before splitting, and it also makes the result reproducible.

### Step 4: split into train / validation / test

The shuffled data is split into three parts.

The current logic uses approximately:

- 65% for training
- the remaining 35% split again into validation and test

So the training split is the largest part, and the validation and test splits share the rest.

### Step 5: write six text files

The script writes six output files:

- `src-train.txt`
- `src-val.txt`
- `src-test.txt`
- `targ-train.txt`
- `targ-val.txt`
- `targ-test.txt`

These are the six documents you asked about.

They are really just the three splits written twice:

- once for the source text
- once for the target regex

So the data is still paired line by line, only stored in separate files.

## What preprocessing means in this project

Preprocessing is the step where raw text is made ready for the model.

In this project it means:

- normalizing placeholders so the same concept always uses the same token
- cleaning unwanted characters and spacing
- splitting text into tokens using whitespace
- optionally creating character-level representations for each token
- adding special tokens like start-of-sequence and end-of-sequence markers
- padding or truncating sequences so every example has a fixed size

### Why this is needed

Neural models expect numbers, not raw strings.
They also work best when each batch has examples of the same shape.

So preprocessing turns text such as:

`find lines containing blue`

into a structured numeric form the model can consume.

## How the indexers are built

After the data is split, the notebook builds vocabularies, also called indexers.

An indexer is a lookup table that assigns a unique integer to each token or character.

This project builds three indexers:

- a source-word indexer for the natural-language input
- a target-word indexer for the regex output
- a character indexer for character-level features

### Special tokens

Each indexer also includes special symbols such as:

- padding token
- unknown token
- begin-of-sequence token
- end-of-sequence token

These symbols help the model know where a sequence starts, where it ends, and how to handle missing or unseen tokens.

### Where the vocab comes from

The vocabulary is built from the training split only.
That keeps the evaluation fair because the model should not peek at validation or test data while building its vocabulary.

### What gets saved

The completed indexers are saved in `indexers.pkl` so the same mappings can be reused later.

That way, training and inference use the exact same token-to-number rules.

## How examples are turned into numbers

Once the indexers exist, each example is converted into numeric sequences.

### Source side

The natural-language description is:

1. split into words,
2. wrapped with start and end tokens,
3. padded or truncated to the maximum sequence length,
4. mapped into integer IDs.

### Target side

The regex is processed the same way:

1. split into tokens,
2. wrapped with start and end tokens,
3. padded or truncated,
4. converted into target IDs.

### Optional character-level features

For each word, the project can also create a character sequence.

This is useful when the model should see smaller pieces of tokens, not just whole words.

Each word is cleaned, wrapped with character-level start/end tokens, and padded or truncated to a fixed character length.

## Typical workflow

If you want to rerun the pipeline from scratch, the flow is:

1. run `preprocess_data.py` to create the split files
2. open `trying.ipynb`
3. run the preprocessing and indexer cells
4. verify the saved `indexers.pkl`
5. map the split data into numeric arrays
6. connect those arrays to the training code

## Important note

The current setup focuses on data preparation and training readiness.
The actual model training step comes after this README’s scope, but the output of this folder is designed to feed directly into it.

## In short

This project takes text/regex pairs, cleans them, splits them, builds vocabularies, and turns them into fixed-size numeric inputs.
That is the full path from raw data to training-ready data.
