import matplotlib.pyplot as plt
import numpy as np


class A:

    def bresenhams_algorithm(self, start_x, start_y, end_x, end_y):
        """
        this function draw line between two cells in coordinate word frame
         return the coordiantes of the new cells
        """
        new_coord_holder = []
        new_coord_holder.append((start_x, start_y))
        if start_x != end_x and start_y != end_y:
            new_x, new_y = 0, 0
            while new_x != end_x and new_y != end_y:
                delta_x = end_x - start_x
                delta_y = end_y - start_y
                m = delta_y / delta_x  # if <1 or not
                P_o = (2 * delta_y) - delta_x
                if P_o < 0:
                    new_x = start_x + 1
                    new_y = start_y
                    P_o += 2 * delta_y

                else:
                    new_x = start_x + 1
                    new_y = start_y + 1
                    P_o += (2 * delta_y) - (2 * delta_x)

                start_x = new_x
                start_y = new_y

                new_coord_holder.append((new_x, new_y))
            new_coord_holder.append((end_x, end_y))
            return new_coord_holder
        elif start_x == end_x:
            small = min(start_y, end_y)
            big = max(start_y, end_y)
            for dist in range(small, big + 1):
                new_coord_holder.append((start_x, dist))
            return new_coord_holder
        elif start_y == end_y:
            small = min(start_x, end_x)
            big = max(start_x, end_x)
            for dist in range(small, big + 1):
                new_coord_holder.append((dist, start_y))
            return new_coord_holder




a=A()
res=a.bresenhams_algorithm(0.,0.,1.,30.)
print(res)
res=np.array(res)
plt.scatter(res[:, 0], res[:, 1], s=20,c='green')
plt.grid()
plt.show()