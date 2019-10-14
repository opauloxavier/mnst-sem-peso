import wisardpkg

ws = wisardpkg.Wisard(4)
ws.train([[0, 0, 1, 1], [0, 1, 1, 1]], ["blue", "dark"])
out = ws.classify([[1, 1, 1, 1]])
print(out)
