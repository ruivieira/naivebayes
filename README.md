# naive

Naive Bayes classifier for Crystal (based on [muatik's classifier](https://github.com/muatik/naive-bayes-classifier)).

## Installation


Add this to your application's `shard.yml`:

```yaml
dependencies:
  gsl:
    github: ruivieira/naivebayes
```


## Usage


```crystal
require "naive"

tokeniser = Naive::Tokeniser.new

newsTrainer = Naive::Trainer.new(tokeniser)

newsSet = [
  {"text" => "not to eat too much is not enough to lose weight", "category" => "health"},
  {"text" => "Russia try to invade Ukraine", "category" => "politics"},
  {"text" => "do not neglect exercise", "category" => "health"},
  {"text" => "Syria is the main issue, Obama says", "category" => "politics"},
  {"text" => "eat to lose weight", "category" => "health"},
  {"text" => "you should not eat much", "category" => "health"},
]

newsSet.each { |news|
  newsTrainer.train(news["text"], news["category"])
}

newsClassifier = Naive::Classifier.new(newsTrainer.data, tokeniser)

classification = newsClassifier.classify("Obama is")

puts classification # => {"health" => 1.6666666666666666e-10, "politics" => 0.083333333333333329}
```

Warning:

- Not fully test
- Pre-release (API will break)
- Not fit for production



## Contributing

1. Fork it ( https://github.com/ruivieira/crystal-gsl/fork )
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request

## Contributors

- [ruivieira](https://github.com/ruivieira) Rui Vieira - creator, maintainer
- [muatik](https://github.com/muatik) muatik - original Python code