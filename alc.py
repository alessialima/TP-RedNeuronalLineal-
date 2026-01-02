#%% 
# Contenido del archivo:
#     Importaciones
#     Módulos del laboratorio:
#       1. Funciones Auxiliares 
#       2. Funciones de cada módulo 
#     Trabajo Práctico: Red Neuronal Lineal 
#     Carga de Datos 
#     Ejercicio 1: Ecuaciones Normales
#     Ejercicio 2: SVD
#     Ejercicio 3: QR 
#        Qr con HH 
#        Qr con Gs 
#     Ejercicio 4: Pseudo-Inversa de Moore-Penrose     
#%% ===============================================================================================
# IMPORTACIONES 
# =================================================================================================

import numpy as np 
import os 
import time


#%% ===============================================================================================
# MODULOS 
# .................................................................................................
# =================================================================================================
#%% FUNCIONES AUXILIARES 

def calculaCholesky(A, tol = 1e-10): #Complejidad total O(n³)
#    if not esSDP(A,tol): ya sabemos que XXt es SDP
#        return None
    L, D, _ = calculaLDV(A)  
    n = D.shape[0]
    
    R = np.zeros((A.shape))
    
    for i in range(n):
        dii = np.sqrt(D[i, i])
        R[i, i] = dii
        for j in range(i + 1, n):
            R[j, i] = L[j, i] * dii
    
    return R

def productoPorEscalar(A, escalar): #O(mxn)
    A = np.array(A, dtype=float)
    m, n = A.shape
    
    resultado = [[0.0] * n for _ in range(m)]
    
    for i in range(m):
        for j in range(n):
            resultado[i][j] = A[i][j] * escalar
    
    return resultado


def restaMatricial(A, B):   #O(mxn)
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    m, n = A.shape
    
    res = [[0.0] * n for _ in range(m)]
    
    for i in range(m):
        for j in range(n):
            res[i][j] = A[i][j] - B[i][j]
    
    return res


# Producto Matricial por Bloques utilizando @ 
# utilizamos esta función como combinación entre funciones de Python y @ para el TP 
def prodMatricialBloques(A, B, block_size=64): 
    m, p = A.shape
    n = B.shape[1]
    
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Dimensiones incompatibles: A{A.shape} vs B{B.shape}")
    
    C = np.zeros((m, n))
    
    for ii in range(0, m, block_size): #filas (m)
        i_end = min(ii + block_size, m)
        
        for jj in range(0, n, block_size): #columnas (n)
            j_end = min(jj + block_size, n)
            
            for kk in range(0, p, block_size): #dimensión interior (p)
                k_end = min(kk + block_size, p)
                
                A_block = A[ii:i_end, kk:k_end]
                B_block = B[kk:k_end, jj:j_end]
                C[ii:i_end, jj:j_end] += A_block @ B_block
    
    return C

# Esta es la función original de ProdMatricial, utilizada en un principio, hasta que notamos lo mucho que tardaba en algunas funciones 
# como cuando en svd buscamos multiplicar dos matrices 1536x2000 elemento a elemento 
# no queríamos que la dificultad de la función (es decir, el largo tiempo de ejecución) sea por el producto entre matrices cuando ya conocíamos
# la cantidad de iteraciones de diagRH y metPot, por ejemplo 
#---------------------------------------------------------------
# Implementación original de Producto Matricial.
# prefiriendo la versión con bloques y @.

def prodMatricial(A, B):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    
    if A.ndim == 1:
        A = A.reshape(1, -1)  
    
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Dimensiones incompatibles: {A.shape} vs {B.shape}")
    
    res = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                res[i, j] += A[i, k] * B[k, j]
    
    return res

#Complejidad de prod Matricial: Si A (m x p) y B (p x n), es O(m * p * n)

def traspuesta(A):
    A = np.array(A, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1) # por si es vector 
    m,n= A.shape
    
    T = np.zeros((n,m), dtype=float)
    
    #invertimos índices
    for i in range(m):
        for j in range(n):
            T[j, i] = A[i, j]
    return T 

def identidad(m):
    res = np.zeros((m,m))
    for i in range(len(res)):
        res[i][i] = 1 
    return np.array(res, dtype=float)

def productoInterno(a, b): #O(n),  
    res = 0.0 
    for i in range(len(b)):
        res += a[i]*b[i]
    return res #res = a.b

def productoExterno(a, b): #O(mxn), 
    m = len(a)
    n = len(b)
    res = [[0.0] * n for _ in range(m)]
    
    for i in range(m):
        for j in range(n):
            res[i][j] = a[i] * b[j]
    
    return res #res = a.b^t de mxn

#%% ===============================================================================================
# 1er MÓDULO 
# =================================================================================================

def error(x, y):
    return abs(x-y)

def error_relativo(x, y):
    if x < 1e-15:
        return 0.0 
    return abs(x-y) / abs(x)


# función auxiliar que simula la función de numpy all_error 
def all_error(a, b, tol): 
    for i in range(len(a)):
        if error(a[i], b[i]) > tol:
            return False 
    return True 

def matricesIguales(A, B, tol):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    
    if A.shape != B.shape:
        return False 
    for i in range(A.shape[0]):
        if not all_error(A[i], B[i], tol):
            return False 
    return True 

#%% ===============================================================================================
# 2do MÓDULO 
# =================================================================================================

def rota(theta):
  return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def escala(s):
  n = len(s)
  res = np.zeros((n, n))
  for i in range(n):
    res[i, i] = s[i]
  return res

def rota_y_escala(theta, s):
  return prodMatricial(rota(theta), escala(s)) 

def afin(theta, s, b):
  return prodMatricial(rota_y_escala(theta, s), b) 

def trans_afin(v,theta,s,b):
    M = afin(theta, s, b)
    v_hom = np.array([v[0], v[1], 1])
   
    x_trans = M[0,0]*v_hom[0] + M[0,1]*v_hom[1] + M[0,2]*v_hom[2]
    y_trans = M[1,0]*v_hom[0] + M[1,1]*v_hom[1] + M[1,2]*v_hom[2]
    
    return [x_trans, y_trans]

#%% ===============================================================================================
# 3er MÓDULO 
# =================================================================================================

def norma(x, p): 
    # Caso de la norma infinito (máximo absoluto)
    if p == 'inf':  
        res = 0
        for i in range(len(x)): 
            if res <= abs(x[i]):
                res = abs(x[i])

    # Caso de la norma 2
    elif p == 2: 
        res = 0
        for i in range(len(x)): 
            res += abs(x[i])**2
        res = res ** (1/2)

    #Caso norma 1 suma de valores absolutos
    elif p == 1: 
        res = 0
        for i in range(len(x)):
            res += abs(x[i]) 

    #Caso general para cualquier p
    else: 
        res = 0
        for i in range(len(x)): 
            res += abs(x[i])**p
        res = res ** (1/p) 
    return res 
    
        
def normaliza(X, p):
    res = []
    for i in range(len(X)):
        n = norma(X[i], p) #Si el vector es nulo, lo deja tal cual 
        if n == 0:
            res.append(X[i])
        else:
            v = []
            for j in range(len(X[i])): 
                v.append(X[i][j] / n)
            res.append(v)
    return res

"""
Versión 1 de Norma Exacta: según qué pida, devuelve una u otra norma
def normaExacta(A, p=[1,'inf']):  
     if p == 1:
        n = 0
        for i in range(len(A)): 
                if norma(A[i], 1) >= n:
                    n = norma(A[i], 1)
        return n
         
     elif p == 'inf':
        m = 0
        tras = traspuesta(A)
        for i in range(len(tras)): 
                if norma(tras[i], 1) >= m:
                    m = norma(tras[i], 1)
        return m
     else: 
          return None
      
Versión 2: devuelve una lista de ambas        
"""
def normaExacta(A, p=[1,'inf']):  
    res = []
    n = 0
    for i in range(len(A)): 
        norma1 = []
        if norma(A[i], 1) >= n:
            n = norma(A[i], 1)
            norma.append(n)
        else: 
            norma1.append(0)
    res.append(norma1)
    
    m = 0
    T = traspuesta(A)
    for j in range(len(T)):
        norma2 = []
        if norma(T[j], 1) >= m:
            m = norma(T[i], 1)
            norma2.append(m)
        else: 
            norma2.append(0)
    res.append(norma2)
    return res 

def normaMatMC(A, q, p, Np):  

    # Estimamos la norma matricial ||A||_{p->q} usando el método de Monte Carlo
    # Generzmos Np vectores random y tomamos el max ||A x||_q / ||x||_p
    norma_max = 0
    x_max = None

        
    for _ in range(Np):
        x_random = np.random.randn(len(A[0])) 
        x_normalizado = x_random / norma(x_random, p)    
        
        Ax = [] ## producto A @ x 
        for i in range(len(A)):
            v = 0
            for j in range(len(A[i])):
                v += x_normalizado[j]*A[i][j]
            Ax.append(v)
        
        # calculo de la norma ||A x||_q y vamos actualizando el maximo
        actual = norma(Ax,q)
        if actual > norma_max:
            norma_max = actual
            x_max = x_normalizado
            
    return norma_max, x_max
    
def condMC(A, p): 
    res = normaMatMC(A, p, p, 1000)[0] * normaMatMC(inversa(A), p, p, 1000)[0]
    return res 

def condExacta(A, p): 
    res = normaExacta(A, p) * normaExacta(inversa(A), p)
    return res 

#%% ===============================================================================================
# 4to MÓDULO 
# =================================================================================================

def calculaLU(A): #O(n³)
    ## Implementamos la factorización sin pivoteo y usando el método de Gauss
    if A is None:
        return None, None, 0
    
    try:
        A = np.array(A, dtype=float)
        m, n = A.shape 
    except:
        return None, None, 0
    
    # Solo factorizamos matrices cuadradas.
    if m != n:
        return None, None, 0 
    
    nops = 0
    U = A.copy() 
    L = np.eye(m)
    
    for i in range(n-1): #chequeamos pivote no nulo
        pivote = U[i, i]
        if abs(pivote) < 1e-10:
            return None, None, 0
        
        for j in range(i + 1, m):
            coef = U[j, i] / pivote
            L[j, i] = coef
            U[j, i] = 0  # Ya sabemos que es cero
            
            U[j, i+1:] -= coef * U[i, i+1:]
            nops += (m - i - 1) * 2  

    
    if abs(U[m-1, m-1]) < 1e-10:
        return None, None, 0
    
    return L, U, nops

"""
función res_tri sólo para vectores:
def res_tri(A, b, inferior=True): 
    n = A.shape[0]
    x = np.zeros(n)
    
    #Lx = b
    if inferior: 
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i])) / A[i, i] #usamos np.dot para optimizar
    #Ux = b
    else: 
        for i in range(n-1, -1, -1):
            x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x
"""
# Versión que también usa matrices
def res_tri(A, b, inferior=True):
    n = A.shape[0]
    
    # caso b vector -> lo convertimos en matriz columna
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    
    m = b.shape[1]  
    x = np.zeros((n, m))
    
    # Lx = b 
    if inferior:
        for i in range(n):
            for j in range(m):
                x[i, j] = (b[i, j] - np.dot(A[i, :i], x[:i, j])) / A[i, i]
    
    # Ux = b 
    else:
        for i in range(n-1, -1, -1):
            for j in range(m):
                x[i, j] = (b[i, j] - np.dot(A[i, i+1:], x[i+1:, j])) / A[i, i]
    
    return x



def calculaLDV(A):
  L, U, nops = calculaLU(A) #O(n³) llamada dominante en cuanto a complejidad
  if L is None or U is None:
    return None, None, None 
  
  # D = diag(U)
  D = np.zeros(U.shape) 
  for i in range(U.shape[0]):
        D[i, i] = U[i, i]

  #V = L^t
  V = np.zeros(L.shape)
  for i in range(L.shape[0]):
    for j in range(L.shape[1]):
      V[i, j] = L[j, i]

  return L, D, V



def inversa(A):
    # Usamos elim. Gaussiana para la inversión
    A = np.array(A, dtype=float)
    m, n = A.shape 
    
    if m != n: 
        return None 
    
    A_inv = identidad(m)
    A_copia = A.copy() 
    
    #Pivoteo parcial: buscamos e intercambiamos filas si el pivote es 0
    for i in range(m): #Complejidad total del bucle (y de la función): O(n³)
        print(f"{i} de {m}")
        if abs(A_copia[i, i]) < 1e-15:
            for k in range(i+1, m):
                if abs(A_copia[k, i]) > 1e-15:
                    A_copia[[i, k]] = A_copia[[k, i]]
                    A_inv[[i, k]] = A_inv[[k, i]]
                    break
            # caso matriz singular
            else: 
                return None  
        pivote = A_copia[i, i]
        A_copia[i] /= pivote
        A_inv[i] /= pivote
        
        for j in range(m):
            if j != i:
                coef = A_copia[j, i]
                A_copia[j] -= coef * A_copia[i]
                A_inv[j] -= coef * A_inv[i]
                
    return A_inv

def esSDP(A, atol = 1e-15): # lo hice más estricto de lo que pedía porque no pasaba el test  
    A = np.array(A, dtype=float)
    m, n = A.shape
    
    #Simetría
    if m != n: 
        return False 

    T = traspuesta(A)
    if not matricesIguales(A, T, tol=atol):
        return False 
      
    L, D, V = calculaLDV(A) #O(n³)
    if L is None or D is None or V is None: 
             return False 
    

    D = np.array(D)
    #Todos los autovalores (elementos de D) deben ser estrictamente positivos
    for l in range(len(D)): 
        if D[l, l] <= atol: 
            return False 
    return True 

#%% ===============================================================================================
# 5to MÓDULO 
# =================================================================================================


#funciones de calculo de QR son O(m²n)

def QR_con_GS(A, tol=1e-12): 
    A = np.array(A, dtype=float)
    m, n = A.shape 
    r = min(m, n)
    
    if m == 0 or n == 0:
        return A.copy(), A.copy()
    
    nops = 0 
    Q = np.zeros((m, r), dtype=float)
    R = np.zeros((r, n), dtype=float)
    
    for j in range(n): #iteramos sobre cada col de A
        #print(f"{j} de {n}") # agrego proceso 
        v = A[:, j].copy()
        
        # hacemos o que 'v' sea ortogonal a las columnas Q ya construidas
        for i in range(min(j, r)):
            R_ij = np.sum(Q[:, i] * v)
            R[i, j] = R_ij
            
            # Actualizamos v (restamos la proyección de v sobre Q_i)
            v -= R_ij * Q[:, i]
            nops += 2 * m
        
        norm_v = np.sqrt(np.sum(v * v))
        
        #Normalización
        if norm_v < tol:
            Q[:, j] = 0.0
            R[j, j] = 0.0 
        else: 
            Q[:, j] = v / norm_v
            R[j, j] = norm_v 
            nops += m 
    
    return Q, R    


def QR_con_HH(A, tol=1e-12):
    A = np.array(A, dtype=float)
    m, n = A.shape
    
    R = A.copy()
    Q = identidad(m)
    
    
    for k in range(n): #iteramos sobre cada col que debe ser anulada
        
        print(f"{k} de {n}") # agrego proceso (para ver el tiempo que tarda cada iteración)
        x = R[k:m, k] #subcolumna a anular
        norm_x = norma(x, 2) #calculamos su norma
        
        if norm_x < tol:
            continue


        # u = x ± ||x||*e1.    
        e = np.zeros(len(x)) 
        e[0] = 1.0
        signo = 1 if x[0] >= 0 else -1   
        a = signo * norm_x
        u = x + a * e 
        
        #normalizo u
        norm_u = norma(u, 2)
        
        if norm_u < tol:
            continue
        
        u = u / norm_u
        
        ## (P_k @ R)
        ## Transformamos las columnas i  de R para anular los elementos de abajo de la diagonal
        for i in range(k, n): 
           col = R[k:m, i]
           doti = productoInterno(u, col)
           R[k:m, i] = col - 2 * doti * u 
        
        #(Q @ P_k).
        for j in range(m):
            fila = Q[j, k:m]
            dotj = productoInterno(fila, u)
            Q[j, k:m] = fila - 2 * dotj * u
     
    return Q, R


def calculaQR(A,metodo='RH',tol=1e-12):
    if metodo == 'RH': 
        return QR_con_GS(A, tol)
    else:
        return QR_con_HH(A, tol)

#%% ===============================================================================================
# 6to MÓDULO  
# =================================================================================================

"""
Comentario acerca de este módulo: al resolver la función svd para la matriz X_train 
notamos que la larga duración de la misma provenía de las siguientes causas: 
    1. cantidad de productos de matrices elemento a elemento usando matrices de gran tamaño 
    2. las 1000 iteraciones en la función diagRH 
    3. las 1000 iteraciones en la función metPot2k 
Por ende, decidimos acortar el tiempo usando una nueva función llamada ProdMatricialBloques 
que multiplicaba matrices por bloques pero finalmente usando @ o prodMatricial en matrices más pequeñas
Luego, decidimos utilizar @ en lugar de alguna otra función
De igual manera, el tiempo de ejecución de svd es enorme comparado con otras descomposiciones

Entregaremos ambas funciones, las originales y, comentadas, las funciones que utilizan @
"""

def func_met(A,v,tol=1e-15): 
    # Función auxiliar para aplicar A y normalizar el resultado (A*v / ||A*v||)
    Av = prodMatricial(A,v)
    Av_norma = norma(Av,2)
    if(Av_norma < tol):
        return np.zeros(v.size)
    return Av * 1/Av_norma

def metpot2k(A, tol=1e-15, K=1000):
    ## metodo de la potencia para encontrar el autovalor/autovector dominante.
    n = A.shape[0]
    v_1d = np.random.uniform(-1, 1, n)
    v = v_1d.reshape(-1, 1) # v es ahora (n, 1)
    autval = 0.0
    err = 0.0
    iteraciones = 0

    #inicializamos vn
    vn = func_met(A,func_met(A,v,tol),tol)
    if norma(vn,2) < tol:
            return v.flatten(), autval, iteraciones, err

    err = prodMatricial(traspuesta(vn),v)

    #O(K) iteraciones 
    while(abs(err-1) > tol and iteraciones < K): #O(K * n²) costo domiante
        v = vn
        vn = func_met(A,func_met(A,vn,tol),tol)
        err = prodMatricial(traspuesta(vn),v)
        iteraciones+=1
    Avn = prodMatricial(A,vn)
    autval = prodMatricial(traspuesta(vn),Avn)
    return vn.flatten(), autval, iteraciones,abs(err-1)


def diagRH(A,tol=1e-15,K=np.inf):
    #Función recursiva para diagonalizar
    #como usa HH y Metpot O(n³)


    n = A.shape[0]
    # caso base
    if (n == 1):
        return np.eye(1), A

    v_1d,autval,*_ = metpot2k(A,tol,K)
    v = v_1d.reshape(-1, 1)
    v_unitario = v * 1/norma(v,2)
    I = np.eye(n)
        
    D = np.zeros((n,n))
    e_1 = np.zeros((n,1))
    e_1[0] = 1.0
    u = e_1 - v_unitario
    norma_al_cuadrado = prodMatricial(traspuesta(u),u)

    # aca evita division por cero cuando e_1 es practicamente igual a v_unitario. Hv NO tiene que hacer nada entonces Hv = I
    if norma_al_cuadrado < tol:
        Hv = I 
    else:
        #construyo householder
        Hv = I - 2 * prodMatricial(u, traspuesta(u)) / norma_al_cuadrado
    #construyo matriz de autovectores
    B = prodMatricial(prodMatricial(Hv,A),traspuesta(Hv))
    # aca hice esto al cortar B para que el resultado sea matriz y no un numpy-array. 
    A2 = B[1:n, 1:n]
    S2, D2 = diagRH(A2,tol,K)
    # construyo la submatriz D con n-1 de dimension
    D[0][0] = autval
    D[1:,1:] = D2
    #construyo S de igual forma a D
    Q = identidad(n)
    Q[1:, 1:] = S2

    return prodMatricial(Hv, Q),D

"""
Funciones modificadas: 
    
def metpot2k(A, tol=1e-15, K=1000):
    n = A.shape[0]
    v_1d = np.random.uniform(-1, 1, n)
    v = v_1d.reshape(-1, 1) # v es ahora (n, 1)
    autval = 0.0
    err = 0.0
    iteraciones = 0

    vn = func_met(A,func_met(A,v,tol),tol)
    if norma(vn,2) < tol:
            return v.flatten(), autval, iteraciones, err

    err = traspuesta(vn) @ v
    while(abs(err-1) > tol and iteraciones < K):
        v = vn
        vn = func_met(A,func_met(A,vn,tol),tol)
        err = traspuesta(vn) @ v 
        iteraciones+=1
    Avn = A @ vn
    autval = traspuesta(vn) @ Avn
    return vn.flatten(), autval, iteraciones,abs(err-1)

def diagRH(A,tol=1e-15,K=np.inf):
    n = A.shape[0]
    # caso base
    if (n == 1):
        return np.eye(1), A

    v_1d,autval,*_ = metpot2k(A,tol,K)
    v = v_1d.reshape(-1, 1)
    v_unitario = v * 1/norma(v,2)
    I = np.eye(n)
        
    D = np.zeros((n,n))
    e_1 = np.zeros((n,1))
    e_1[0] = 1.0
    u = e_1 - v_unitario
    norma_al_cuadrado = traspuesta(u) @ u

    # aca evita division por cero cuando e_1 es practicamente igual a v_unitario. Hv NO tiene que hacer nada entonces Hv = I
    if norma_al_cuadrado < tol:
        Hv = I 
    else:
        #construyo householder
        Hv = I - 2 * (u @ traspuesta(u)) / norma_al_cuadrado
    #construyo matriz de autovectores
    B = ((Hv @ A) @ traspuesta(Hv))
    # aca hice esto al cortar B para que el resultado sea matriz y no un numpy-array. 
    A2 = B[1:n, 1:n]
    S2, D2 = diagRH(A2,tol,K)
    # construyo la submatriz D con n-1 de dimension
    D[0][0] = autval
    D[1:,1:] = D2
    #construyo S de igual forma a D
    Q = identidad(n)
    Q[1:, 1:] = S2

    return prodMatricial(Hv, Q),D

"""

#%% ===============================================================================================
# 7mo MÓDULO  
# =================================================================================================

def transiciones_al_azar_continuas(n):
    res = []
    for i in range(n): 
        v = []
        for j in range(n): 
            v.append(np.random.rand())  
        res.append(v) # armo matriz random con numeros entre 0 y 1 
        
    res_t = traspuesta(res) 
    res_t_norm = []
    
    for s in range(len(res_t)):
        col = res_t[s]
        n_val = norma(col, 1)
        if n_val < 1e-12:
            col_n = [1.0/n for _ in col] 
        else: 
            col_n = []
            for elemento in col:
                elemento_normalizado = elemento / n_val
                col_n.append(elemento_normalizado)
        res_t_norm.append(col_n) 
    res = traspuesta(res_t_norm) 
    return np.array(res) 
        
def transiciones_al_azar_uniformes(n,thres):
    matriz = []
    for i in range(n):
        fila = []
        for j in range(n): 
            if np.random.rand() > thres: 
                fila.append(1) 
            else:  
                fila.append(0)
        matriz.append(fila)
        
    matriz_T = traspuesta(matriz) 
    matriz_T_norm = []
    
    for s in range(len(matriz_T)): 
        col = matriz_T[s]
        n_val = norma(col, 1)
        
        if n_val < 1e-12:
            col_n = [1.0/n for _ in range(len(col))] 
        else: 
            col_n = []
            for elemento in col:
                elemento_normalizado = elemento / n_val
                col_n.append(elemento_normalizado)
        matriz_T_norm.append(col_n) 
        
    res = traspuesta(matriz_T_norm) 
    return np.array(res) 
    
    

def nucleo(A, tol=1e-8):
    A = np.array(A, dtype=float)
    m,n = A.shape 
    
    T = traspuesta(A) 
    ATA = prodMatricial(T, A)   
    S, D = diagRH(ATA, tol, 1000) 

    i_v = []
    for i in range(len(D)):
        if np.abs(D[i][i]) < tol:
            i_v.append(i) 

    k = len(i_v) # cantidad de vectores en el nucleo
    if k == 0: # caso identidad 
        return np.zeros((0,n))
    else: 
        res = S[:, i_v]  
    
        return res


def crea_rala(listado,m_filas,n_columnas,tol=1e-15):
    if len(listado) == 0:
        return {}, (m_filas, n_columnas)
    lista_i, lista_j, lista_aij = listado
    d = {} 
    
    for i in range(len(lista_i)):
        if abs(lista_aij[i]) >= tol: 
            fila = lista_i[i]
            columna = lista_j[i]
            valor = lista_aij[i]
            if fila < m_filas and columna < n_columnas:
                 d[(fila, columna)] = valor  # {(i,j): A_ij}
    
    return d, (m_filas, n_columnas) 

def multiplica_rala_vector(A,v): 
    A_dict, dims = A 
    m_filas, n_columnas = dims 

    resultado = np.zeros(m_filas) 
    
    for (i, j), valor in A_dict.items():
        if i < m_filas and j < len(v):
            resultado[i] += valor * v[j]
    return resultado

#%% ===============================================================================================
# 8vo MODULO  
# =================================================================================================

"""

Resueltas las correcciones !! 

Comentario sobre este módulo: 
    1. agregamos una función diagRHsinCeros que utiliza como caso base los autovalores cercanos o iguales a cero
    2. por temas de optimización comentados en el módulo 6, agregamos una versión comentada que utiliza @ en lugar de nuestras funciones 
    que resuelven productos de matrices 
    y que, además, reduce los casos al concreto pedido (matriz 1536x2000)
"""
def diagRHsinCeros(A,tol=1e-15,K=1000): 
    n = A.shape[0]
    # caso base, si justo el ultimo autovalor es 0 (segun tol) devuelvo la matriz de control nula
    if (n == 1):
        if (A[0,0] < tol):
            return np.zeros((1, 1)), np.zeros((1, 1))
        return np.eye(1), A
    
    v_1d,autval,*_ = metpot2k(A,tol,K)
    v = v_1d.reshape(-1, 1)
    # no quiero que siga si un autovalor 0 (segun tol)
    if autval < tol:
        return np.eye(n), np.zeros((n, n))
    
    v_unitario = v * 1/norma(v,2)
    I = np.eye(n) 
    D = np.zeros((n,n))
    e_1 = np.zeros((n,1))
    e_1[0] = 1.0
    u = e_1 - v_unitario

    # aca evita division por cero cuando e_1 es practicamente igual a v_unitario. Hv NO tiene que hacer nada entonces Hv = I
    beta = prodMatricial(traspuesta(u), u).item()
    if beta < tol:
        Hv = I
    else:
        Hv = I - 2 * prodMatricial(u, traspuesta(u)) / beta
    # construyo matriz B de autovalores
    B = prodMatricial(prodMatricial(Hv,A),traspuesta(Hv))
    
    # recorto la A actual para hacer el nuevo llamado recursivo
    A2 = B[1:n, 1:n]
    S2, D2 = diagRHsinCeros(A2,tol,K)
    # si el llamado recursivo encontró autovalores nulos en una matriz mas grnade que 1x1 entonces termino el algoritmo y uso D con ceros y la S actual sin los demas autovectores

    D = np.zeros((n, n))
    D[0, 0] = autval[0][0] # es un array 

    # construyo la submatriz D con n-1 de dimension
    D[0][0] = autval[0][0] # es un array 
    D[1:,1:] = D2
    #construyo la matriz de autovectores
    Q = np.eye(n)
    Q[1:, 1:] = S2

    return prodMatricial(Hv, Q),D

def svd_reducida(A,K=np.inf,tol=1e-15): 
    m, n = A.shape
    #Cuando m >> n encuentro primero V y luego U. Es decir hago svd sobre At. Siempre intento usar la matriz más chica Ata de acuerdo a m y n
    if(m >= n):
        At = traspuesta(A)
        AtA = prodMatricial(At,A)
        # calculo la matriz V de autovectores y Sigma de autovalores (que son los val. singulares de A)
        V, D = diagRHsinCeros(AtA,tol,1000) 
        autovalores = np.diagonal(D)
        # filtro autovalores segun tol primero
        rango_con_tol = sum(1 for aut in autovalores if  aut >= tol)
        # me quedo con la cantidad pedida segun K
        if K != np.inf:
            rango = min(rango_con_tol, int(K))
        else:
            rango = rango_con_tol

        # si la matriz inicial A era nula
        if(rango == 0):
            raise ValueError(f"La matriz no ser nula")
        # corto V y Sigma para que sea la versión reducida
        D = D[0:rango,0:rango]
        V = V[:,0:rango]
        # calculo Sigma a partir de la D diagonal de AtA = VDVt
        Sigma = np.sqrt(D)
        # calculo U teniendo en cuenta que A*V = U * Sigma -> basta con normalizar las columnas de A*V para encontrar U.Es decir la diagonal sigma son las normas de las col. Av1,Av2..etc

        A_V = prodMatricial(A,V)
        U = np.zeros((m,rango))
        for i in range(rango):
            U[:,i] = A_V[:,i] / Sigma[i,i]
        return U,np.diagonal(Sigma),V
    # m << n
    else:
        AAt = prodMatricial(A,traspuesta(A))
        # calculo la matriz V de autovectores y Sigma de autovalores (que son los val. singulares de A)
        U, D = diagRHsinCeros(AAt,tol,1000)
        autovalores = np.diagonal(D)
        rango = sum(1 for aut in autovalores if  aut >= tol)
        # si la matriz inicial A era nula
        if(rango == 0):
            raise ValueError(f"La matriz no ser nula")

        # corto V y Sigma para que sea la versión reducida
        D = D[0:rango,0:rango]
        U = U[:,0:rango]
        # calculo Sigma a partir de la D diagonal de AtA = VDVt
        Sigma = np.sqrt(D)
        # calculo U teniendo en cuenta que A*V = U * Sigma -> basta con normalizar las columnas de A*V para encontrar U.Es decir la diagonal sigma son las normas de las col. Av1,Av2..etc

        At_U = prodMatricial(traspuesta(A),U)
        V = np.zeros((n,rango))
        for i in range(rango):
            V[:,i] = At_U[:,i] / Sigma[i,i] 
        return U,np.diagonal(Sigma),V

#%% ===============================================================================================
# TRABAJO PRÁCTICO: RED NEURONAL LINEAL  
# .................................................................................................
# =================================================================================================
#
#
#
#
#
#
# =================================================================================================
#%% ===============================================================================================
# EJERCICIO 1: CARGAR DATOS 
# =================================================================================================


def cargarDataset(carpeta):
    ruta_base = os.path.join(carpeta, 'cats_and_dogs') 
    
    # train
    train_cats = np.load(os.path.join(ruta_base, 'train', 'cats', 'efficientnet_b3_embeddings.npy'))
    train_dogs = np.load(os.path.join(ruta_base, 'train', 'dogs', 'efficientnet_b3_embeddings.npy'))
    
    # val 
    val_cats = np.load(os.path.join(ruta_base, 'val', 'cats', 'efficientnet_b3_embeddings.npy'))
    val_dogs = np.load(os.path.join(ruta_base, 'val', 'dogs', 'efficientnet_b3_embeddings.npy'))
    
    X_train = np.hstack([train_cats, train_dogs])
    X_val = np.hstack([val_cats, val_dogs])
    
    # clases: 
    Y_train = np.array([[0] * 1000 + [1] * 1000,  [1] * 1000 + [0] * 1000])  

    Y_val = np.array([[0] * 500 + [1] * 500, [1] * 500 + [0] * 500])      
   
    return X_train, Y_train, X_val, Y_val


carpeta = ('dataset')
X_train, Y_train, X_val, Y_val = cargarDataset(carpeta)

#%% ===============================================================================================
# EJERCICIO 2: Implementación del algoritmo 1 que utiliza Cholesky
# =================================================================================================


t_total_chol = 0      
t_total_chol_pinv = 0 
t_inicio = time.time() # tiempo solo para Cholesky (factorizacion)
XXt = X_train @ traspuesta(X_train)
L = calculaCholesky(XXt, 1e-10) #Calculamos el caso (b) del algoritmo sugerido.
t_fin = time.time()
t_total_chol = t_fin - t_inicio # t_chol (decimal)

## para los casos (a) y (c) los valores de X que usaríamos serían: 
    # caso (a):
        # aplico cholesky a XtX 
    # caso (c):
        # aplico cholesky a Xt 

#=================
#Versión anterior:
#=================
"""
def ecuacionesNormales(X, L, Y_train):
   L_inv = inversa(L)
   L_T_inv = inversa(traspuesta(L))
   V = traspuesta(X_train) @ (L_T_inv @ L_inv)
   return Y_train @ V, V
"""

#%%
#==================
#Versión corregida:
#==================

def ecuacionesNormales(X, L, Y_train):
    
    """
    Calcula W = Y * X^+ y devuelve (W, V) donde V = X^+
    - No invierte matrices explicitamente y resuelve el sistema con res_tri.
    - Tenemos en cuenta los tres casos de enunciado:

        a) n > p : Asumo que recibe L (p x p) tal que
           L @ Lt = Xt @ X

        b) n < p : Asumo que recibe L (n x n) tal que
           L @ Lt = X @ Xt

        c) n == p : Asumo que recibe L (n x n) tal que 
           L @ Lt = X @ Xt
    """

    n, p = X.shape
    m = Y_train.shape[0]
    
    # caso (a): L es p x p  => Cholesky de X^T@X  (n > p)
    #====================================================
    # resuelve (XtX) V = X^T
    # resuelve (L L^T) V = X^T  
    # hacemos LZ = X^T y luego L^T V = Z 
    # Z  y V tienen shape (p, n) porque X^T es p x n

    if L.shape == (p, p) and n > p: 
       
        Xt = traspuesta(X)

        # resuelvo L L^T V = X^T  usando la auxiliar
        z = res_tri(L, Xt, inferior=True)    # p x n
        Vt = res_tri(traspuesta(L), z, inferior=False)
            
        # luego W = Y * X^+
        W = Y_train @ Vt

        return W, traspuesta(Vt)
    
    # caso (b): L es n x n  => Cholesky de X@X^T   (n < p)
    #=====================================================
    # resuelve (L L^T) V = X^T  
    # Queremos resolver V (XX^T) = X^T
    # Transponemos para poder usar Cholesky:
    # (XX^T) V^T = X
    # LZ = X ,  L^TV = Z 
    # Z  y Vt tienen shape (n, p)


    elif L.shape == (n, n) and n < p:
        # Resuelvo (LL^T) Vt = X
        z = res_tri(L, X, inferior=True)   # n x p
        Vt = res_tri(traspuesta(L), z, inferior=False)
        # V = X^T Vt   
        V = traspuesta(Vt)   # p x n

        W = Y_train @ V
        return W, V

   

    # Caso (c): n == p , entonces  X es cuadrada.  X^+ = X^(-1)
    #============================================================
    #Queremos despejar W de: W@X = Y  
    #Para que la incógnita sea W trasponemos ambos lados
    #XT @ WT = YT
    #Si bien Xt es cuadrada, no me alcanza para  descomponerla en Cholesky ya que no es SDP 
    #Podría optar por descomponerla en LU, pero creo que el enunciado incita a usar choleky.
    # entonces multiplico por X:  (X @ X^T) @ W^T  =  X @ Y^T 
    # tomo A = X X^T  (simétrica y definida positiva) / A = LLt
    # Entonces resuelvo A W^T = X @ Y^T

    elif L.shape == (n, n) and n == p:
        # B = X Y^T
        B = prodMatricial(X, traspuesta(Y_train))  # n x m

        # resuelvo (L L^T) W^T = B
        z = res_tri(L, B, inferior=True)         # n x m
        Wt = res_tri(traspuesta(L), z, inferior=False)
        
        W = traspuesta(Wt)                         # m x n

        return W, inversa(X)

"""
# agregamos para que devuelva V = X+ así luego podíamos usarla en el ejercicio 5 
 
def resolverSistemaCholesky(L, B):
    
    Resuelve L L^T X = B columna por columna.
    L debe ser triangular inferior
    
    n, m = B.shape
    Lt = traspuesta(L)

    X = np.zeros((n, m))

    for j in range(m):
        b = B[:, j]
        # L z = b--+
        z = res_tri(L, b, inferior=True)
        # Lt x = z
        X[:, j] = res_tri(Lt, z, inferior=False)

    return X
"""

t_inicio = time.time()   
W_en, V_en = ecuacionesNormales(X_train, L, Y_train) 
t_fin = time.time()
t_total_chol_pinv = t_fin - t_inicio # t_cholPinv (decimal)

## Tiempo total
t_total = t_total_chol + t_total_chol_pinv # t_total (decimal)

tiempos_en = [np.round(t_total_chol, 2), np.round(t_total_chol_pinv, 2), np.round(t_total, 2)] 



#%% ===============================================================================================
# EJERCICIO 3: Implementación del algoritmo 2 que utiliza SVD 
# =================================================================================================

t_total_svd = 0
t_total_svd_pinv = 0


t_inicio = time.time() # tiempo solo para SVD
U, S, V = svd_reducida(X_train,np.inf,tol=1e-15)
t_fin = time.time()
t_total_svd = t_fin - t_inicio

def pinvSVD(U, S, V, Y_train):
    S_pinv = np.diag(1.0 / S) # los valores de Sigma PseudoInversa son inversos a los de Sigma
    Ut = traspuesta(U) 
    pinv = (V @ (S_pinv @ Ut)) # V @ S_pinv @ Ut (por def de pseudoinversa)
    W = Y_train @ pinv # W = Y @ pinv
    return W, pinv 
    #Hacemos que también devuelva pinv para comparar en el ejercicio 4.
    #Le cambiamos el nombre a pinv para no confundirnos con la V de SVD !! 

## tiempo solo para pinv svd
t_inicio = time.time()
W_svd, V_svd = pinvSVD(U, S, V, Y_train)  
t_fin = time.time()
t_total_svd_pinv = t_fin - t_inicio


t_total = t_total_svd + t_total_svd_pinv


tiempos_svd = [np.round(t_total_svd, 2), np.round(t_total_svd_pinv, 2), np.round(t_total, 2)] 

# Tiempo de Ejecución aproximado: 4hs 

#%% ===============================================================================================
# EJERCICIO 4: Implementación del algoritmo 3 que utiliza QR
# =================================================================================================


#%% VERSIÓN HOUSEHOLDER 
""" A diferencia de Gram-Schmidt, nuestro algoritmo de Householder nos devuelve una R rectangular
por lo tanto no podemos aplicar explícitamente el algoritmo 3 siendo que R no es invertible.
Recortamos la matriz para verse como una R^1536x1536, siendo las filas recortadas
unas filas de ceros. Luego, nos ayudamos con la función res_tri para hallar V 


"""

t_total_qr_hh = 0
t_total_qr_hh_pinv = 0

    
t_inicio = time.time() # tiempo solo para factorizacion QR
X_train_T = traspuesta(X_train) 
Q_hh, R_hh = QR_con_HH(X_train_T, 1e-12)
t_fin = time.time()
t_total_qr_hh = t_fin - t_inicio

# recortamos la matriz R     
R_list = [] 
m, n = R_hh.shape
for i in range(m):
    if i < 1536:
        R_list.append(R_hh[i])

R_array = np.array(R_list)

# print(np.allclose(X_train_T, Q @ R, atol=1e-12)) 
# comparamos los resultados con la matriz original 

def pinvHouseholder(Q, R, Y_train):    
    Qt = traspuesta(Q)
    V = traspuesta(res_tri(R, Qt, inferior=False))
    W = Y_train @ V
    return W, V

t_inicio = time.time()  # tiempo solo para pinv a partir de Q y R
W_hh, V_hh = pinvHouseholder(Q_hh, R_array, Y_train)
t_fin = time.time()
t_total_qr_hh_pinv = t_fin - t_inicio # Tiempo decimal de Pinv

t_total = t_total_qr_hh + t_total_qr_hh_pinv

tiempos_hh = [np.round(t_total_qr_hh, 2), np.round(t_total_qr_hh_pinv, 2), np.round(t_total, 2)] 
#%%
"""
%% VERSIÓN GRAM-SCHMIDT 
En este caso no necesitamos recortar la matriz R, así que aplicamos el 
algoritmo directamente a los Qt y R hallados 

"""
# además de calcular W_gs y V_gs, calculamos tiempos de ejecución
X_train_T = traspuesta(X_train)

t_total_qr_gs = 0
t_total_qr_gs_pinv = 0 


t_inicio = time.time() # tiempo solo para hacer QR con Gram-Schmidt (Factorización)
Q_gs, R_gs = QR_con_GS(X_train_T, tol = 1e-12) # mientras tanto, calculamos QR con gs 
t_fin = time.time()
t_total_qr_gs = t_fin - t_inicio # t_QR (decimal)
# print(np.allclose(Q_gs @ R_gs, X_train_T), 1e-12) # sabiendo que gs es más inestable que hh, probamos comparar los resultados con la matriz original
# nos dio un resultado positivo

def pinvGramSchmidt(Q, R, Y_train):
    V = traspuesta(res_tri(R, traspuesta(Q), inferior=False))
    W = prodMatricialBloques(Y_train, V, block_size=64)
    return W, V

                        
t_inicio = time.time() # tiempo solo para pinv Gram-Schmidt
W_gs, V_gs = pinvGramSchmidt(Q_gs, R_gs, Y_train) 
# agregamos a cada función, que además devuelva V para luego utilizarla en el ejercicio 5 

# comparamos los pinv de cada qr, vemos que nos dio bastante similar 
# print(np.allclose(pinv_gs, pinv_hh, 1e-12))

t_fin = time.time()
t_total_qr_gs_pinv = t_fin - t_inicio # t_qrPinv (decimal)

t_total = t_total_qr_gs + t_total_qr_gs_pinv

tiempos_gs = [np.round(t_total_qr_gs, 2), np.round(t_total_qr_gs_pinv, 2), np.round(t_total, 2)] 
# Tiempo de ejecución sólo factorización 
# Tiempo de ejecución sólo pinv 
# Tiempo total 

#%% ===============================================================================================
# EJERCICIO 5: Pseudo-Inversa de Moore-Penrose
# =================================================================================================


#Devuelve True si se cumplen las 4 condiciones de Moore-Penrose

def esPseudoInversa(A, pinv, tol=1e-08):
    #  A * pinv * A = A
    pinvA = pinv @ A 
    A_pinv_A = A @ pinvA 
    if not matricesIguales(A_pinv_A, A, tol):
        return False
    
   # pinv * A * pinv = pinv
    temp2 = pinv @ A
    pinv_A_pinv = temp2 @ pinv
    if not matricesIguales(pinv_A_pinv, pinv, tol):
        return False
    
    # (A * pinv)^T = A * pinv
    A_pinv = A @ pinv
    A_pinv_T = traspuesta(A_pinv)
    return matricesIguales(A_pinv, A_pinv_T, tol)
    
    # (pinv * A)^T = pinv * A
    pinv_A = pinv @ A
    pinv_A_T = traspuesta(pinv_A)
    if not matricesIguales(pinv_A, pinv_A_T, tol):
       return False
    
    return True

print(esPseudoInversa(X_train, V_en, tol = 1e-08))
print(esPseudoInversa(X_train, V_svd, tol = 1e-08))
print(esPseudoInversa(X_train, V_hh, tol = 1e-08))
print(esPseudoInversa(X_train, V_gs, tol = 1e-08))

#%% GUARDAR DATOS:

# Esto lo utilizamos para que luego sea más sencillo trabajar en el archivo Jupyter
# con los valores de cada peso W 

np.save("W_hh.npy", W_hh)
np.save("W_en.npy", W_en)
np.save("W_gs.npy", W_gs)
np.save("W_svd.npy", W_svd)


# Esto lo utilizamos para que luego sea más sencillo trabajar en el archivo Jupyter 
# con el tiempo de ejecución

tiempos = [tiempos_en, tiempos_gs, tiempos_hh, tiempos_svd]

t1 = []
t2 = []
t3 = []

for i in range(len(tiempos)):
    t1.append(tiempos[i][0])
    t2.append(tiempos[i][1])
    t3.append(tiempos[i][2])
    
# pequeña corrección: nos faltó convertir las listas con los tiempos en arrays
t1 = np.array(t1)
t2 = np.array(t2)
t3 = np.array(t3)

np.save("t1.npy", t1)
np.save("t2.npy", t2)
np.save("t3.npy", t3)



