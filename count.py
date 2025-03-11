import csv

def contar_entradas_csv(caminho_arquivo):
    try:
        with open(caminho_arquivo, mode='r', newline='', encoding='utf-8') as arquivo:
            leitor_csv = csv.reader(arquivo)
            # Pula o cabeçalho
            next(leitor_csv, None)
            # Conta as entradas
            contador = sum(1 for linha in leitor_csv)
            return contador
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {caminho_arquivo}")
        return None
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return None

# Exemplo de uso
caminho_do_arquivo = 'AI_Human.csv'
numero_de_entradas = contar_entradas_csv(caminho_do_arquivo)

if numero_de_entradas is not None:
    print(f"O arquivo CSV tem {numero_de_entradas} entradas.")