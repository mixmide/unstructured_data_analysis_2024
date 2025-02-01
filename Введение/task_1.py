def is_valid_email(email: str) -> bool:
    '''
    Адрес электронной почты должен удовлетворять следующим требованиями:
    1) не должно быть пробелов
    2) должен быть ровно один символ собаки
    3) слева и справа от собаки должны быть символы
    4) слева от собаки может быть что угодно
    5) справа от собаки обязательно должна быть точка и эта точка не должна быть последним символом
    6) не должно быть несколько точек подряд
    '''

    if ' ' in email:
        return False

    if "@" not in email:
        return False

    split_by_dog = email.split('@')

    if len(split_by_dog) != 2 or split_by_dog[0] == '' or split_by_dog[1] == '':
        return False

    second_part = split_by_dog[1]

    if '.' not in second_part or '..' in second_part or second_part[-1] == '.':
        return False

    return True

email = input()

print(is_valid_email(email))
