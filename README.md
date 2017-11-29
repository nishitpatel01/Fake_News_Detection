# fake_news_detection_dev
Fake News Detection in Python

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


	Note that we do not provide the full-text verdict report in this current version of the dataset,
	but you can use the following command to access the full verdict report and links to the source documents:
	wget http://www.politifact.com//api/v/2/statement/[ID]/?format=json

	======================================================================
	The original sources retain the copyright of the data.

	Note that there are absolutely no guarantees with this data,
	and we provide this dataset "as is",
	but you are welcome to report the issues of the preliminary version
	of this data.

	You are allowed to use this dataset for research purposes only.

	For more question about the dataset, please contact:
	William Wang, william@cs.ucsb.edu

	v1.0 04/23/2017

	Below is the process of building features and classifiers for prediction:
	
	<mxfile userAgent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:57.0) Gecko/20100101 Firefox/57.0" version="7.7.7" editor="www.draw.io" type="github"><diagram id="a172fee5-a165-84c1-2139-96accb38b3d5" name="Page-1">7Zvdc+MmEMD/Gj8mow9/Pia+S/pw18k0N3e9pw6WsMwcEirCidO/vosEsgx2LSuS7cZOJhNpgRXw2xULiJ4/jVePHKWLryzEtOc54arnf+p5nusPJvBPSt4KyWjkF4KIk1BlWgueyT9YCR0lXZIQZxsZBWNUkHRTGLAkwYHYkCHO2etmtjmjm09NUYQtwXOAqC39QUKxKKTjgbOW/4ZJtNBPdh2VMkPBr4izZaKe1/P8ef5TJMdI61L5swUK2WtF5H/u+VPOmCiu4tUUU9m3utuKcg87Ust6c5yIOgW8osALokusazykUPR+zkADdB0KioTh30tZqfvvmIcoQWsBXEXy/xPHKWcBzjKSRFoLPLlQVORRjRZvuqPzrsKyMi4kvy6IwM9p8chXMC2QLURMVfKcUDpllPG8rB8iPJ4HIM8EZ79wJWUYjPFsrhrxrJ7mqvsHFBMq7VK3xL9/wVwQoH9HSZRAyowJwWJIiDgKCXSl1p2wBMsHampu2SapAq92UnBLtuAzmMVY8DfIogr42rCUu/jq9nVte662mEXF7kZKhpS5R6XmNXK4UNS3W4C/xQIMSDgJ76RTwV1AEfANNrkUBXBo+dTe9lcaONjSPi3jmCJBXjbVb2u0esITI7nxqu7tu5vdW3qrVpGxJQ+wKlX1ln2KBoYigXiEhaUoZ1A2uxaWvoUluQFrjDObDqXwWlRmmeIcEluGXbnTFvcxvaxr93D7Nf1j3IJ/jN/vHzu6Da+I+BMkzu1A3f2UBW4d11f3T5gTqDCWHJwDHA1ql1t0xZRO5XsmO2/Sku9ZcNvzvYmFfArDlKzLA0ZiybHthIeNYzvc6N3D24bjRdIUleEELJZGWTGidzmkOzHGq7oOOWzBIbUdVPB8m9+QcH7lszOeOCof1+Lzg/HQ+46DKyFNaOCektC2qH/3mKaC3v3dX3d4OtVY5PX3DCF1x6LB5GhjkWsHgucff5wPYLcZYFNR6Z4dAB5cATeeyTUGbCrqEvDwCrg2YPvN2hCwqahLwCML8O8IuuDmHr3JQMeZ5kjnBCh0EvWYi1wHR0HHinqMt2oZBB0l6LEn8l9YRDJBZFj6B464XLlkyZXX7mHwuMDsafjzMk0Zl14OkwkB3eI5X1GwIBCfXqmtveyU1LTeKjXBggVSjvaoVtTh8hPOguLqik4VGJ4UnT1xvwYqdaca/aZ7CpYivXfafqCieV4B15lqOJN2AFuKOgTcwmbexQA2ZwiNAVuKOgR82GrQh1m5K3dsdReb411dViN3j6IWWXW3sHMxuExFZTzUAa7ulmnOHddep2jqXV3ishddLgWX3xYuU1GXuLr7jOLccZlbfY1xmYq6xGWvvfwXrg8bZwx1u9/7JrQUtcdKfyx4ga5lhQfmB5FN44yhV8+1oFPRWyVbKjNkB1R4svF5LlwUGhvbQndLOOduC9Zg1tQWTEV1baEJLu9icVmDWVNcpqIucXX3Lca547Ii+6a47IGxO1w1Yk79yTSJ8xMhVVL6UMAXNMP0iWVEyN3ByuEAKhPuywMflX0DdeTDPlggmNyxyB92l6XFyRVHS/JqRNCqgBIAD81/eMSI/+V64xX83aZJtM+A3vfVoLllroLAioX1t1iYubHeZOuhf7kxjLk83DjktBSNOws5+3aY8Y3DlTdEsTTwZJaledd/tHvnAdEMW6Z50CZlrb3Eeh+4zgfyd+vOZf7Ts08wQdos/23pnWG+zrXRVZzHc7Z4Txsblv3Dwic1Sw1RtihZNd74kNcJ1LeSJG9/9v4fmx/my2LUP7u3Dtyuz0EW2deHTf3P/wI=</diagram></mxfile>
	
