class Tokeniser:

    def __init__(self, string):
        self.string = string
        self.char2int = {}
        self.int2char = {}
        token = 0
        for char in (string):
            if(char not in self.char2int):
                self.char2int[char] = token
                self.int2char[token] = char
                token += 1; 
        self.vocab_size = len(self.char2int)
    
    def encode(self, substring):
        list = []
        for char in substring:
            list.append(self.char2int[char])
        return list 

    def decode(self, tokens):
        string = ""
        for num in tokens:
            string += (self.int2char[num])
        return string 

tokenizer = Tokeniser("hello world")

ids = tokenizer.encode("hello")
print(ids)

text = tokenizer.decode(ids)
print(text)

print(tokenizer.vocab_size)

print(tokenizer.decode(tokenizer.encode("hello")) == "hello")