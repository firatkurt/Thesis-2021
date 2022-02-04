

def a(idf) :
    idf =  ['5','2', 'C','D', '+']
    result = []
    for id in idf:
        print(id)
        if id.isnumeric():
            result.append(int(id))
        elif id == 'C':
            result.pop()
        elif id == 'D':
            result.append( 2 * result[-1])
        elif id == '+':
            result.append(result[-1] + result[-2])
    sums = sum(result)
    print(sums)
    return sums


def b():
    test = '{[]}'
    result =0
    for c in test:
        if c == '{':
            result = result+1
        if c == '[':
            result = result +1

    print(result)

b()