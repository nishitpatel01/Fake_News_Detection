# Fake News Detection

Fake News Detection in Python

This project is part of CS410:Text Information System course. We have used various natural language processing and machine learning libaries from python. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them:

Dataet used:
	-- The data source used for this project is from LIAR dataset which has 3 files with .tsv format for test, train and validation. Below is some description about the data files used for this project.
	
	
	-- LIAR: A BENCHMARK DATASET FOR FAKE NEWS DETECTION

	William Yang Wang, "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection, to appear in Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017), short paper, Vancouver, BC, Canada, July 30-August 4, ACL.
	=====================================================================
	Description of the TSV format:

	Column 1: the ID of the statement ([ID].json).
	Column 2: the label. (Label class contains: True, Mostly-true, Half-true, Barely-true, FALSE, Pants-fire)
	Column 3: the statement.
	Column 4: the subject(s).
	Column 5: the speaker.
	Column 6: the speaker's job title.
	Column 7: the state info.
	Column 8: the party affiliation.
	Column 9-13: the total credit history count, including the current statement.
	9: barely true counts.
	10: false counts.
	11: half true counts.
	12: mostly true counts.
	13: pants on fire counts.
	Column 14: the context (venue / location of the speech or statement).



```
Give examples
```

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With



## Contributing



## Versioning


## License



## Acknowledgments
