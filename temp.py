import shelve
from matplotlib import pyplot as plt

with shelve.open("validation") as shelve_db:
    counter = 0
    for key in shelve_db:
        # print(key)
        # item = shelve_db[key]
        # img = item["char_image"]
        # plt.imshow(img)
        counter += 1

    print(counter)

