module Naive
  class NotSeen < Exception
  end

  class Trainer
    getter data

    def initialize(@tokeniser : Tokeniser)
      @data = TrainedData.new
    end

    # enhances trained data using the given text and class
    def train(text : String, className : String)
      @data.increaseClass(className)

      tokens = @tokeniser.tokenise(text)
      tokens.each { |token|
        token = @tokeniser.remove_stop_words(token)
        token = @tokeniser.remove_punctuation(token)
        @data.increaseToken(token, className)
      }
    end
  end

  class TrainedData
    getter docCountOfClasses, frequencies

    def initialize
      @docCountOfClasses = Hash(String, Int32).new
      @frequencies = Hash(String, Hash(String, Int32)).new
    end

    def increaseClass(className : String, byAmount : Int = 1)
      @docCountOfClasses[className] = @docCountOfClasses.fetch(className, 0) + 1
    end

    def increaseToken(token : String, className : String, byAmount : Int = 1)
      if !@frequencies.keys.any? { |key| key == token }
        @frequencies[token] = {} of String => Int32
      end

      @frequencies[token][className] = @frequencies[token].fetch(className, 0) + 1
    end

    # returns all documents count
    def getDocCount
      return @docCountOfClasses.values.sum
    end

    # returns the names of the available classes as list
    def getClasses
      return @docCountOfClasses.keys
    end

    # returns document count of the class.
    # If class is not available, it returns Nil
    def getClassDocCount(className : String)
      return @docCountOfClasses.fetch(className, Nil)
    end

    def getFrequency(token : String, className : String) : Int32 | Nil
      foundToken = @frequencies[token]?

      if foundToken.nil?
        raise NotSeen.new
      else
        frequency = foundToken[className]?
        return frequency
      end
    end
  end

  class Tokeniser
    def initialize(@stop_words : Array(String) = [] of String, @signs : Array(String) = ["?!#%&"])
    end

    def tokenise(text : String)
      return text.downcase.split(" ")
    end

    def remove_stop_words(token : String)
      @stop_words.includes?(token) ? "stop_word" : token
    end

    def remove_punctuation(token : String)
      @signs.each { |sign|
        sign.each_char { |s| token = token.sub(s, "") }
      }
      return token
    end
  end

  class Classifier
    def initialize(@trainedData : TrainedData, @tokeniser : Tokeniser)
      @data = trainedData
      @defaultProb = 0.000000001
    end

    def getTokenProb(token : String, className : String) : Float64
      # p(token|Class_i)
      classDocumentCount = @data.getClassDocCount(className)
      # if the token is not seen in the training set, so not indexed,
      # then we return None not to include it into calculations.
      tokenFrequency = @data.getFrequency(token, className)
      if tokenFrequency.nil?
        return @defaultProb
      else
        probablity = tokenFrequency.to_f64 / classDocumentCount.as(Int32).to_f64
        return probablity
      end
    end

    def classify(text : String)
      documentCount = @data.getDocCount
      classes = @data.getClasses

      # only unique tokens
      tokens = @tokeniser.tokenise(text).uniq
      probsOfClasses = {} of String => Float64

      classes.each { |className|
        # we are calculating the probablity of seeing each token
        # in the text of this class
        # P(Token_1|Class_i)
        tokensProbs = tokens.map { |t| getTokenProb(t, className) }

        # calculating the probablity of seeing the the set of tokens
        # in the text of this class
        # P(Token_1|Class_i) * P(Token_2|Class_i) * ... * P(Token_n|Class_i)
        tokensSetProb = tokensProbs.reduce { |acc, p| acc * p }
        probsOfClasses[className] = tokensSetProb * getPrior(className)
      }
      return probsOfClasses
    end

    def getPrior(className : String)
      return @data.getClassDocCount(className).as(Int32).to_f64 / @data.getDocCount.to_f64
    end
  end
end
