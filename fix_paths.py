import codecs
with codecs.open('tests/run_tests.py', 'r', encoding='utf-8') as f:
    content = f.read()

target1 = 'LIB_CANDIDATES = [os.path.join(ADAPTQ_DIR, "build_clean", lib_name), os.path.join(ADAPTQ_DIR, "build_release", lib_name)]'
target2 = 'LIB_CANDIDATES = [os.path.join(ADAPTQ_DIR, "build_clean", lib_name), os.path.join(ADAPTQ_DIR, "build_release", \nlib_name)]'
target3 = 'LIB_CANDIDATES = [os.path.join(ADAPTQ_DIR, "build_clean", lib_name), os.path.join(ADAPTQ_DIR, "build_release", \r\nlib_name)]'

replacement = 'LIB_CANDIDATES = [os.path.join(ADAPTQ_DIR, "build", lib_name), os.path.join(ADAPTQ_DIR, "build", "Release", lib_name), os.path.join(ADAPTQ_DIR, "build_clean", lib_name), os.path.join(ADAPTQ_DIR, "build_release", lib_name)]'

content = content.replace(target1, replacement).replace(target2, replacement).replace(target3, replacement)

with codecs.open('tests/run_tests.py', 'w', encoding='utf-8') as f:
    f.write(content)
