def matmul_biasses(A, B, C, bias):
    m, p, n = A.shape[0], A.shape[1], B.shape[1]
    
    for i in range(m):
        for j in range(n):
            for k in range(p):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C

def matmul_goto(self, fused_HW, Ck2, OC, fused_cols, kernel_matrix, output):
    for jc in range(0, fused_HW, self.nc): # Bucle L1
        nc_eff = min(self.nc, fused_HW - jc)
        for pc in range(0, Ck2, self.kc): # L2
            kc_eff = min(self.kc, Ck2 - pc)
            # Pack B
            self.Bc = fused_cols[pc:pc+kc_eff, jc:jc+nc_eff]
            for ic in range(0, OC, self.mc): # L3
                mc_eff = min(self.mc, OC - ic)
                # Pack A
                self.Ac = kernel_matrix[ic:ic+mc_eff, pc:pc+kc_eff]
                for jr in range(0, nc_eff, self.nr): # L4
                    nr_eff = min(self.nr, nc_eff - jr)
                    for ir in range(0, mc_eff, self.mr): # L5
                        mr_eff = min(self.mr, mc_eff - ir)
                        # Micro-kernel
                        for kk in range(0,kc_eff,1):
                            for ii in range(0,mr_eff,1):
                                for jj in range(0,nr_eff,1):
                                    output[ic+ir+ii, jc+jr+jj] += (self.Ac[ir+ii, kk] * self.Bc[kk, jr+jj])
    return output

def matmul_goto_np(self, fused_HW, Ck2, OC, fused_cols, kernel_matrix, output):
    for jc in range(0, fused_HW, self.nc): # Bucle L1
        for pc in range(0, Ck2, self.kc): # L2
            # Pack B
            self.Bc = fused_cols[pc:pc+self.kc,jc:jc+self.nc]
            for ic in range(0, OC, self.mc): # L3
                # Pack A
                self.Ac = kernel_matrix[ic:ic+self.mc,pc:pc+self.kc]
                for jr in range(0, self.nc, self.nr): # L4
                    for ir in range(0, self.mc, self.mr): # L5
                        # Micro-kernel
                        output[ic+ir:ic+ir+self.mr,jc+jr:jc+jr+self.nr] += self.Ac[ir:ir+self.mr,0:self.kc] @ self.Bc[0:self.kc,jr:jr+self.nr]
    
    return output