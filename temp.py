import os

def countlines(rootdir, total_lines=0, header=True, begin_start=None,
               code_only=True):
    def _get_new_lines(source):
        total = len(source)
        i = 0
        while i < len(source):
            line = source[i]
            trimline = line.lstrip(" ")

            if trimline.startswith('#') or trimline == '':
                total -= 1
            elif '"""' in trimline:  # docstring begin
                if trimline.count('"""') == 2:  # docstring end on same line
                    total -= 1
                    i += 1
                    continue
                doc_start = i
                i += 1
                while '"""' not in source[i]:  # docstring end
                    i += 1
                doc_end = i
                total -= (doc_end - doc_start + 1)
            i += 1
        return total

    if header:
        print('{:>10} |{:>10} | {:<20}'.format('ADDED', 'TOTAL', 'FILE'))
        print('{:->11}|{:->11}|{:->20}'.format('', '', ''))

    for name in os.listdir(rootdir):
        file = os.path.join(rootdir, name)
        if os.path.isfile(file) and file.endswith('.py'):
            with open(file, 'r') as f:
                source = f.readlines()

            if code_only:
                new_lines = _get_new_lines(source)
            else:
                new_lines = len(source)
            total_lines += new_lines

            if begin_start is not None:
                reldir_of_file = '.' + file.replace(begin_start, '')
            else:
                reldir_of_file = '.' + file.replace(rootdir, '')

            print('{:>10} |{:>10} | {:<20}'.format(
                    new_lines, total_lines, reldir_of_file))

    for file in os.listdir(rootdir):
        file = os.path.join(rootdir, file)
        if os.path.isdir(file):
            total_lines = countlines(file, total_lines, header=False,
                                     begin_start=rootdir, code_only=code_only)
    return total_lines

countlines("/Users/louiepietroni/Desktop/Desktop - Louie’s MacBook Pro/Cambridge/II/Project/temp-new/", code_only=True)