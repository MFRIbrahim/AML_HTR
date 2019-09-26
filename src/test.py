from util import Replacer

replace = Replacer()

with open("../dataset/sentences.txt", "r") as fp:
    for line in fp:
        if line.startswith("#"):
            continue
        line = line.strip().split(" ")[9]
        # print(f"'''{line}''' -> '''{replace(line)}'''")


def assertEq(real, expected):
    if real != expected:
        raise SystemExit(f"Got: '{real}', Expected: '{expected}'")


assertEq(replace('house|built|earlier|than|1918|.|"'), 'house built earlier than 1918."')
assertEq(replace('two|inter-comm.|rec.|,|mod.|k.|and|b.|,|sep.|W.C.'), 'two inter-comm. rec., mod. k. and b., sep. W.C.')
assertEq(replace('Palestinian|teacher|,|R.|Eleazar|(|3rd|Cent.|)|:'), 'Palestinian teacher, R. Eleazar (3rd Cent.):')
assertEq(replace('Mortar|(|the|CHARAUSES|in|Hebrew|)|that|binds|all|the|"|hard-'), 'Mortar (the CHARAUSES in Hebrew) that binds all the "hard-')
assertEq(replace('facts|"|together|and|explains|them|,|represents|the|Oral'), 'facts" together and explains them, represents the Oral')
