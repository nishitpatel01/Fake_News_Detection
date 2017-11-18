# fake_news_detection_dev
Fake News Detection in Python

	-- The data source used for this project is from LIAR dataset which has 3 files with .tsv format for test, train and validation. 
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

		LIAR: A BENCHMARK DATASET FOR FAKE NEWS DETECTION


