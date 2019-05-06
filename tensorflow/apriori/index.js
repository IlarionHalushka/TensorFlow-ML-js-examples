const Apriori = require('apriori');

new Apriori.Algorithm(0.15, 0.6, false).showAnalysisResultFromFile('./data.csv');
