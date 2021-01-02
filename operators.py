import numpy as np
import sko.operators.mutation as mutation

def crossover_ox(self):
    # Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom

    for i in range(0, self.size_pop, 2):
        chrom_cpys = [self.Chrom[i, :].copy(), self.Chrom[i + 1, :].copy()]
        s = np.random.randint(0, self.len_chrom)
        t = np.random.randint(0, self.len_chrom)
        if s > t:
            s, t = t, s
        for pInd in range(2):
            pInd_1 = pInd ^ 1
            m_set = set(chrom_cpys[pInd][s:t])
            list_fill = [None] * self.len_chrom
            list_fill[s:t] = chrom_cpys[pInd][s:t]
            fill_ind = 0
            for gInd in range(0, self.len_chrom):
                gene_bit = chrom_cpys[pInd_1][gInd]
                if gene_bit not in m_set:
                    while s <= fill_ind < t:
                        fill_ind += 1
                    list_fill[fill_ind] = gene_bit
                    fill_ind += 1
            self.Chrom[i + pInd, :] = list_fill
    return self.Chrom

def cool_down(self):
    self.T = self.T * 0.92

def get_new_x(self, x):
    x_new = x.copy()
    x_new = mutation.reverse(x_new)
    return x_new
