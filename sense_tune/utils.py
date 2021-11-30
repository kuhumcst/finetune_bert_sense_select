
def process(s):
    if type(s) != str:
        print(s)
    if '||' in s:
        return s.split('||')
    else:
        return [s]
