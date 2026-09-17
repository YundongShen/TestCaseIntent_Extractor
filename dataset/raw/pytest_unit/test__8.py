# test_drawMake.py
import pytest
from unittest.mock import Mock
from draw import drawMake  # Substitua 'your_module' pelo nome do seu módulo

def test_drawMake_winner_found():
    # Mock do cursor e banco de dados
    cursor = Mock()
    db = Mock()
    cursor.fetchall.return_value = [
        (1, 'User1', '12345678901', '1', '2', '3', '4', '5')
    ]
    numsDraw = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    
    result = drawMake(cursor, db, numsDraw)
    
    cursor.execute.assert_called_with('UPDATE userdata SET winner = 1 WHERE cpf = 12345678901 ;')
    db.commit.assert_called_once()
    assert result == numsDraw
# test_draw.py
import pytest
from unittest.mock import Mock
import random

def test_drawMake_max_draws_reached():
    cursor = Mock()
    db = Mock()
    
    # Mocking the fetchall method to return no winners
    cursor.fetchall.return_value = [
        (1, 'User1', '12345678901', 90, 80, 70, 60, 100),
        (2, 'User2', '12345678902', 60, 70, 80, 90, 100)
    ]
    
    # Predefinindo números sorteados para garantir que não haja ganhadores
    numsDraw = [1, 2, 3, 4, 5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30]

   

    result = drawMake(cursor, db, numsDraw)

    # Verifica se o número máximo de sorteios foi atingido
    assert len(result) == 30  # Devem ser 25 sorteios adicionais

    # Verifica se não houve ganhadores

# Para rodar os testes, execute o comando pytest no terminal:
# pytest test_draw.py


def test_validateCPFTest():
    assert validateCPFTest("12345678909") == True  # CPF inválido
    assert validateCPFTest("52998224725") == True   # CPF válido
    assert validateCPFTest("123") == False          # CPF inválido (muito curto)
    assert validateCPFTest("52998224726") == False  # CPF inválido (último dígito errado)


def validateCPFTest(cpf):
                    # VARIÁVEIS CÁLCULO CPF
                    j = 10
                    k = 11
                    sumJ = 0
                    sumK = 0
                    accept = 0
                    
                        
                       
                    if cpf.isdigit() and len(cpf) == 11:  # Confere se o CPF é composto só por números e tem 11 de tamanho
                            for i in range(9):  # Verifica o primeiro número verificador
                                sumJ += int(cpf[i]) * j  # Soma os 9 primeiros números do CPF * a variável j
                                j -= 1
                            for i in range(10):  # Verifica o segundo número verificador
                                sumK += int(cpf[i]) * k  # Soma os 10 primeiros números do CPF * a variável k
                                k -= 1

                            remainderJ = sumJ % 11  # Resto da soma dos números dividido por 11
                            remainderK = sumK % 11  # Resto da soma dos números dividido por 11

                            if remainderJ == 0 or remainderJ == 1:  # Se o resto for 1 ou 0
                                if int(cpf[9]) == 0:  # O 10º número tem que ser 0
                                    accept += 1  # Variável para saber se o número verificador está certo
                            elif int(cpf[9]) == 11 - remainderJ:  # Se o resto não for 1 ou 0, o número verificador tem que ser 11 - o resto
                                accept += 1  # Variável para saber se o número verificador está certo
                            if remainderK == 0 or remainderK == 1:  # Se o resto for 1 ou 0
                                if int(cpf[10]) == 0:  # O 11º número tem que ser 0
                                    accept += 1  # Variável para saber se o número verificador está certo
                            elif int(cpf[10]) == 11 - remainderK:  # Se o resto não for 1 ou 0, o número verificador tem que ser 11 - o resto
                                accept += 1  # Variável para saber se o número verificador está certo
                            
                            if accept == 2:  # Se a variável for 2 então o CPF é válido
                                print("CPF VALIDO")
                                return True
                            else: return False
                    else: return False

                             




if __name__ == "__main__":
    pytest.main()