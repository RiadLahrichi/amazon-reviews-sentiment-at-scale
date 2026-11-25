# DistilBERT vs logistic regression

Trained on 79,635 reviews, tested on 20,042 reviews (same sample for both models). DistilBERT: 2 epochs, reviews cut at 128 tokens, on Tesla T4.

| model | macro_f1 | f1_negative | f1_positive | accuracy | training_seconds |
|---|---|---|---|---|---|
| Logistic regression (TF-IDF) | 0.8785 | 0.8098 | 0.9472 | 0.9173 | 5 |
| DistilBERT (fine-tuned) | 0.9199 | 0.8738 | 0.9661 | 0.9465 | 495 |

DistilBERT is right where logistic regression is wrong on 1,074 test reviews, and wrong where it is right on 489.

Reviews DistilBERT gets right and logistic regression gets wrong:

- negative: "Faltaron las barritas metálicas para sujetar los botones traseros. Ahora no puedo usarlo."
- positive: "Game"
- negative: "This game is not any fun at all first off all the new Pokémon are pretty lame and after about 5 hours of playing I have seen none of the original 150 Pokémon, the game would 100 times better if it onl..."
- positive: "For the money this keyboard is unbeatable. It's small, and well made. I had some trouble getting it to work with my PC initially, for some reason if I plug it into certain USB ports it causes my PC to..."
- positive: "A lot of people complain about difficulty getting the controllers into the cradle, but it's not any harder than plugging it in with a cord. I do wish it had an extra port for the motion controller, bu..."
