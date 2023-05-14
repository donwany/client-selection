```shell
  seltype == 'rand':
      np.random.choice(self.size, p=self.ratio, size=self.sample_ratio, replace=True)
  seltype == 'pow-d':
      np.random.choice(self.size, p=self.ratio, size=self.powd, replace=False)
      
      - sort local_loss from high to low
      - use those losses in the next round
  eg. clients = [6, 8, 5, 4, 0, 9, 1, 2, 7]
      next round = [5, 0, 2, 7, 9, 8, 4, 1, 6]
```