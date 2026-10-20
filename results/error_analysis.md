Sample: 159,424 training and 39,897 test reviews. Macro F1: 0.891

### Error rate by star rating

|   rating |   reviews | error_rate   |
|---------:|----------:|:-------------|
|        1 |      5590 | 6.2%         |
|        2 |      2327 | 17.4%        |
|        4 |      5655 | 16.0%        |
|        5 |     26325 | 4.9%         |

### Strongest words and pairs

| most negative   | most positive    |
|:----------------|:-----------------|
| not             | great            |
| not worth       | love             |
| worst           | perfect          |
| disappointed    | awesome          |
| terrible        | excellent        |
| not recommend   | amazing          |
| useless         | best             |
| horrible        | loves            |
| boring          | perfectly        |
| broke           | not bad          |
| not work        | happy            |
| disappointing   | fun              |
| not happy       | easy             |
| awful           | not disappointed |
| return          | nice             |

### Most confident mistakes

Negative reviews (1-2 stars) the model was surest were positive:

- 1 stars, P(positive) = 1.00: "great product"
- 1 stars, P(positive) = 1.00: "Highly recommend"
- 1 stars, P(positive) = 1.00: "good"
- 2 stars, P(positive) = 1.00: "Good."
- 2 stars, P(positive) = 1.00: "good"

Positive reviews (4-5 stars) the model was surest were negative:

- 5 stars, P(positive) = 0.00: "Horrible!"
- 4 stars, P(positive) = 0.00: "Does not work too well. Return-Refund"
- 5 stars, P(positive) = 0.00: "The microphone stopped working in less than a month, absolutely trash headset"
- 5 stars, P(positive) = 0.00: "Wasn’t worth the money"
- 5 stars, P(positive) = 0.00: "The condition of the product was great but the game itself is glitchy and poorly done, wouldn't recommend buying even at the low price. Bad game not worth it"

### Keeping vs removing negation words

| negation words   |   macro F1 (all test reviews) |   accuracy on reviews with a negation |
|:-----------------|------------------------------:|--------------------------------------:|
| kept             |                         0.891 |                                 0.88  |
| removed          |                         0.875 |                                 0.875 |

40.4% of the test reviews contain no, nor or not.
