import sys
import math

class Address:
    def __init__(self, address: str) -> None:
        """
            Recebe um endereço de memória hexadecimal,
            e o converte para binário, inteiro e salva hexadecimal também.
        """
        address = address.replace("\n", "").replace("0x", "")

        self.hexadecimal = address
        self.binary = str(bin(int(address, 16))[2:].zfill(32))
        self.integer = int(self.binary, 2)

    def get_parts(self, num_indexs, num_offsets_block) -> tuple:
        """
            Retorna a tag do endereço, de acordo com o número de bits,
            mas de forma já endereçada em vetor
        """

        tag = self.binary[:-max(num_indexs+num_offsets_block, 1)]

        return tag, self.binary[len(tag):-max(num_offsets_block, 1)]


class CacheMemory:
    def __init__(self, cache_size, line_size, group_size):
        self.cache_size = cache_size #Tamanho da cache, em bytes.
        self.line_size = line_size #Tamanho da linha da cache, em bytes.
        self.group_size = group_size #Número de elementos no grupos, em unidades.
        self.num_lines = cache_size // line_size #Número de linhas da cache.
        self.num_groups_on_cache = self.num_lines // group_size #Número de linhas que um grupo suporta.
        self.cache = [[None] * self.group_size for _ in range(self.num_groups_on_cache)]

        if math.log2(self.num_lines).is_integer() is False:
            raise Exception("O número de linhas deve ser potência de 2!")
        
        if math.log2(self.num_groups_on_cache).is_integer() is False:
            raise Exception("O número de grupos deve ser potência de 2!")

        """
            Quantidade de linhas na cache representa a quantidade de bits
            necessários para representar o index e utilizar no mapeamento direto
        """
        
        self.num_bits_index = int(math.log2(self.num_groups_on_cache))

        """
            Quantidade de bits separados para ser o OFFSET de bloco, possibilitando
            diferenciar qual das palavras do bloco é a desejada
        """
        self.offset_line = int(math.log2(self.line_size))

        self.hits = 0
        self.misses = 0

    def read(self, address):
        raise NotImplementedError()
    
    def print(self):
        """
            Imprime o estado atual da cache no arquivo de saída.
        """
        with open("output.txt", "a") as file:
            file.write("================\n")
            file.write("IDX V ** ADDR **\n")

            for i, line in enumerate(self.cache):
                for j, block in enumerate(line):
                    crr = str(i*self.group_size + j)
                    
                    index = "0"*(3-len(crr)) + crr

                    valid = 1 if block is not None else 0

                    if block is None:
                        address = " "*10
                    else:
                        address = hex(int(block.binary[:-self.offset_line], 2))
                        address = "0x" + address.replace("0x", "").zfill(8).upper()

                    file.write(f"{index} {valid} {address}\n")

    def print_hits_and_misses(self):
        with open("output.txt", "a") as file:
            file.write("\n")
            file.write(f"#hits: {self.hits}\n")
            file.write(f"#miss: {self.misses}\n")

"""
    Classe para representar a implementação de uma Cache apenas com mapeamento direto.
"""
class DirectMappingCache(CacheMemory):
    def __init__(self, cache_size, line_size, group_size) -> None:
        super().__init__(cache_size, line_size, group_size)

    """
        Cache_size --> Tamanho da cache em bytes.
        Line_size --> Tamanho da linha em bytes.
        Group_size --> Tamanho do grupo, em unidades.
    """

    """
        Função de leitura de bloco da memória (Busca do dado na memória)
    """
    def read(self, address: Address) -> None:
        tag, index = address.get_parts(self.num_bits_index, self.offset_line)
        
        if len(index) > 0:
            line = int(index, 2)
        else:
            line = 0
        
        """
            Cache miss (Bloco vazio, bit de validade seria 0).
        """
        if self.cache[line][0] is None:
            self.cache[line][0] = address
            self.misses += 1
            return
        
        """
            Buscando a TAG do bloco armazenado no index, obtido através do 'address', recebido como parâmetro.
        """
        tag_cache = self.cache[line][0].get_parts(self.num_bits_index, self.offset_line)[0]

        """
            Bloco não está vazio, então compara a TAG para ver se o bloco está presente.
        """
        if tag_cache == tag:
            self.hits += 1 #TAG do endereço é igual a TAG da cache, HIT!
        else:
            self.cache[line][0] = address #Tags diferentes, dado não está presente, miss.
            self.misses += 1
    
"""
    Classe para representar a implementação de uma cache com associatividade (N-ways / N-vias).
"""
class SetAssociativeCache(CacheMemory):
    def __init__(self, cache_size, line_size, group_size):
        super().__init__(cache_size, line_size, group_size)
        """
            Cache_size --> Tamanho da cache em bytes.
            Line_size --> Tamanho da linha em bytes.
            group_size --> Tamanho do bloco de palavras lidos da memória, em bytes.
        """

        """
            Declarando uma matriz para representar o LRE de cada um dos blocos de um conjunto.
        """
        self.less_recently_used = [list() for _ in range(self.num_groups_on_cache)] 

    """
        Função de leitura de bloco da memória (Busca do dado na memória)
    """
    def read(self, address):
        tag, index = address.get_parts(self.num_bits_index, self.offset_line)
        
        if len(index) > 0:
            group = int(index, 2)
        else:
            group = 0

        """
            Busca do bloco de memória dentro do conjunto, localizado através do index
            encontrado no 'address' recebido como parâmetro.
        """
        
        for i, block in enumerate(self.cache[group]):
            if block is None: #Bit de validade == 0, posição i no conjunto está vazia, aloca o bloco, atualiza o LRE e registra MISS.
                self.cache[group][i] = address
                self.less_recently_used[group] += [i]
                self.misses += 1
                return
            """
                Se o conjunto não está vazio, compara a tag com a tag do primeiro bloco
                pertencente ao conjunto, e verifica se é o dado desejado.
            """
            tag_cache = block.get_parts(self.num_bits_index, self.offset_line)[0]

            if tag_cache == tag: #As tags são iguais, bloco encontrado, atualiza o LRE e registra HIT.
                self.hits += 1
                
                self.less_recently_used[group].remove(i)
                self.less_recently_used[group].append(i)
                return

        """
            Após não ter encontrado nenhum espaço vazio dentro do conjunto e
            não ter encontrado o bloco comparando as tags, remove bloco que foi usado a mais tempo
            e adiciona o novo bloco, atualiza o LRE e registra MISS.
        """
        less_block = self.less_recently_used[group].pop(0)
        self.cache[group][less_block] = address
        self.less_recently_used[group].append(less_block)
        self.misses += 1


args = sys.argv

"""
    Função de erro, necessitando que o usuário forneça adequadamente no input
    os tamanhos da cache, da linha e do bloco, todos em bytes.
"""
if len(args) < 4:
    raise Exception("Está faltando argumentos!")

cache_size = int(args[1]) #Tamanho lido do parâmetro 1 do input (tamanho da cache em bytes).
line_size = int(args[2]) #Tamanho lido do parâmetro 2 do input (tamanho da linha em bytes).
group_size = int(args[3]) #Tamanho lido do parâmetro 3 do input (tamanho do bloco em bytes).
input_file = args[4] #Nome do arquivo de entrada, lido do parâmetro 4 do input.

"""
    Sessão robustez do simulador de Cache.
"""
if cache_size % line_size != 0 or line_size > cache_size:
    raise Exception("O tamanho da cache deve ser múltiplo do tamanho da linha!")

if (cache_size // line_size) % group_size != 0:
    raise Exception("O número de linhas deve ser múltiplo do número de grupos!")

if math.log2(cache_size).is_integer() is False:
    raise Exception("O tamanho da cache deve ser potência de 2!")

if math.log2(line_size).is_integer() is False:
    raise Exception("O tamanho da linha deve ser potência de 2!")

if math.log2(group_size).is_integer() is False:
    raise Exception("O número de elementos no grupos deve ser potência de 2!")

"""
    Leitura dos endereços de memória à serem lidos.
"""
addresses = []
with open(input_file, "r") as file:
    for line in file:
        address = Address(line)
        addresses.append(address)

"""
    Declarando o tipo da Cache, se baseando no tamanho da linha e dos blocos.
    1 == group_size indica mapeamento direto na cache, visto que cada bloco cabe exatamente em 1 linha, não tendo conjuntos.
"""
if 1 == group_size:
    cache = DirectMappingCache(cache_size, line_size, group_size)
else:
    cache = SetAssociativeCache(cache_size, line_size, group_size)

# Limpar o arquivo de saída
with open("output.txt", "w") as file: pass

for address in addresses:
    cache.read(address)
    cache.print()

cache.print_hits_and_misses()
