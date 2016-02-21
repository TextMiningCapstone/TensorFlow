import random
def generate_data(filename, size):
    with open(filename+'.en', 'w') as en_writer:
        with open(filename+'.fr', 'w') as fr_writer:
            for i in range(size):
                length = int(random.random()*25)+1
                word = []
                for j in range(length):
                    ch = chr(ord('a')+int(26*random.random()))
                    word.append(str(ch))
                en_writer.write(' '.join(word)+'\n')
                word.reverse()
                fr_writer.write(' '.join(word)+'\n')

if __name__ == '__main__':
    generate_data('data/train', 2000000)
    generate_data('data/dev', 500)
