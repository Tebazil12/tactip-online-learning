import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cbook as cbook
img = plt.imread("/home/lizzie/Pictures/stimulus-flower.png")

print(img.shape)
img_cropped = img[:, 0:int(img.shape[0]/2), :]
# imageFile = cbook.get_sample_data("/home/lizzie/Pictures/stimulus-flower.png")
# img = plt.imread(imageFile)

fig, ax = plt.subplots()

plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))


ax.set_axisbelow(True)

# ax.imshow(img)
x = range(70)
f_size = 126
f_y_offset = -5.2
ax.imshow(img_cropped,zorder=0,  extent=[-f_size/2, 0, 0+f_y_offset, f_size+f_y_offset], alpha=0.5)
ax.plot(x, x, '--', linewidth=5, color='firebrick')


plt.show()
