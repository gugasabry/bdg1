import sqlite3
import cv2
import numpy as np
import os
import urllib.request
from datetime import datetime

# =====================================================================
# 1. CONFIGURAÇÕES INICIAIS, DIRETÓRIOS E MODELO XML
# =====================================================================
SENHA_ADMIN = "admin"
PASTA_ACESSOS = "acessos_g1"
ARQUIVO_XML = "haarcascade_frontalface_default.xml"

os.makedirs(PASTA_ACESSOS, exist_ok=True)

# Baixa automaticamente o arquivo XML de detecção facial se não existir
if not os.path.exists(ARQUIVO_XML):
    print("Baixando arquivo de detecção facial (haarcascade)...")
    url_xml = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url_xml, ARQUIVO_XML)
    print("Download concluído!")

detector_rosto = cv2.CascadeClassifier(ARQUIVO_XML)

# Conexão com o SGBD SQLite
conexao = sqlite3.connect("sistema_acesso_g1.db")
cursor = conexao.cursor()

# Ativa a Integridade Referencial (Chave Estrangeira) no SQLite
cursor.execute("PRAGMA foreign_keys = ON;")

# =====================================================================
# 2. DEFINIÇÃO DE DADOS - DDL (CREATE TABLE, PK, FK e CASCADE)
# =====================================================================
cursor.execute("""
CREATE TABLE IF NOT EXISTS pessoas_autorizadas (
    id_pessoa INTEGER NOT NULL,
    nome VARCHAR(100) UNIQUE NOT NULL,
    foto_arquivo VARCHAR(150) NOT NULL,
    CONSTRAINT pk_pessoas PRIMARY KEY (id_pessoa)
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS registros_acessos (
    id_registro INTEGER NOT NULL,
    id_pessoa INTEGER NOT NULL,
    data_hora TIMESTAMP NOT NULL,
    CONSTRAINT pk_registros PRIMARY KEY (id_registro),
    CONSTRAINT fk_registro_pessoa FOREIGN KEY (id_pessoa)
        REFERENCES pessoas_autorizadas (id_pessoa)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);
""")
conexao.commit()


# =====================================================================
# 3. FUNÇÕES AUXILIARES (WEBCAM, TREINAMENTO E LISTAGEM)
# =====================================================================
def capturar_foto_webcam(nome_pessoa):
    """Abre a webcam e salva a foto (.png) ao apertar ESPAÇO."""
    camera = cv2.VideoCapture(0)
    caminho_foto = os.path.join(PASTA_ACESSOS, f"{nome_pessoa}.png")
    foto_salva = False

    print("\n[WEBCAM] Olhe para a câmera e aperte ESPAÇO para salvar a foto...")

    while True:
        sucesso, frame = camera.read()
        if not sucesso:
            print("Erro ao acessar a webcam.")
            break

        tela = frame.copy()
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostos = detector_rosto.detectMultiScale(cinza, scaleFactor=1.3, minNeighbors=5)

        for (x, y, largura, altura) in rostos:
            cv2.rectangle(tela, (x, y), (x + largura, y + altura), (255, 200, 0), 2)

        cv2.putText(tela, "Aperte ESPACO para salvar a foto", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Cadastro Facial", tela)

        if cv2.waitKey(1) & 0xFF == 32:
            cv2.imwrite(caminho_foto, frame)
            print(f"[SUCESSO] Foto salva em: {caminho_foto}")
            foto_salva = True
            break

    camera.release()
    cv2.destroyAllWindows()
    return caminho_foto if foto_salva else None


def treinar_reconhecedor():
    """Lê as fotos cadastradas no banco e treina o reconhecedor do OpenCV."""
    cursor.execute("SELECT id_pessoa, nome, foto_arquivo FROM pessoas_autorizadas;")
    cadastrados = cursor.fetchall()

    rostos_treino = []
    ids_treino = []
    mapa_nomes = {}

    for id_p, nome, foto_path in cadastrados:
        if os.path.exists(foto_path):
            imagem = cv2.imread(foto_path, cv2.IMREAD_GRAYSCALE)
            if imagem is None:
                continue

            faces = detector_rosto.detectMultiScale(imagem, scaleFactor=1.2, minNeighbors=5)
            if len(faces) > 0:
                x, y, l, a = faces[0]
                rosto_recortado = cv2.resize(imagem[y:y+a, x:x+l], (200, 200))
            else:
                rosto_recortado = cv2.resize(imagem, (200, 200))

            rostos_treino.append(rosto_recortado)
            ids_treino.append(id_p)
            mapa_nomes[id_p] = nome

    if len(rostos_treino) > 0:
        reconhecedor = cv2.face.LBPHFaceRecognizer_create()
        reconhecedor.train(rostos_treino, np.array(ids_treino))
        return reconhecedor, mapa_nomes

    return None, {}


def listar_usuarios():
    """Consulta (DQL) todos os usuários da Tabela Pai."""
    cursor.execute("""
        SELECT id_pessoa, nome, foto_arquivo
        FROM pessoas_autorizadas
        ORDER BY nome ASC;
    """)
    usuarios = cursor.fetchall()

    if not usuarios:
        print("\nNenhuma pessoa cadastrada no momento.")
        return []

    print("\n--- TABELA PAI: PESSOAS AUTORIZADAS ---")
    for id_p, nome, foto in usuarios:
        print(f"ID (PK): {id_p} | Nome: {nome} | Arquivo: {foto}")
    return usuarios


# =====================================================================
# 4. OPERAÇÕES DO MENU - DML E CONSULTAS (DQL)
# =====================================================================
def cadastrar_acesso():
    nome = input("\nDigite o nome completo da pessoa: ").strip()
    if not nome:
        print("O nome não pode ficar vazio (Restrição NOT NULL).")
        return

    caminho_foto = capturar_foto_webcam(nome)
    if caminho_foto:
        try:
            cursor.execute("""
                INSERT INTO pessoas_autorizadas (nome, foto_arquivo)
                VALUES (?, ?);
            """, (nome, caminho_foto))
            conexao.commit()
            print(f"Usuário '{nome}' cadastrado com sucesso!")
        except sqlite3.IntegrityError:
            print("Erro: Este nome já está cadastrado (Restrição UNIQUE).")


def editar_acesso():
    usuarios = listar_usuarios()
    if not usuarios:
        return

    id_escolhido = input("\nDigite o ID do usuário que deseja editar: ").strip()

    cursor.execute("""
        SELECT id_pessoa, nome, foto_arquivo
        FROM pessoas_autorizadas
        WHERE id_pessoa = ?;
    """, (id_escolhido,))
    usuario = cursor.fetchone()

    if not usuario:
        print("Usuário não encontrado.")
        return

    id_p, nome_atual, foto_atual = usuario
    print(f"\nEditando: {nome_atual}")
    print("1 - Alterar nome")
    print("2 - Tirar uma nova foto")
    opcao = input("Escolha uma opção: ").strip()

    if opcao == "1":
        novo_nome = input("Digite o novo nome completo: ").strip()
        if novo_nome:
            nova_foto = os.path.join(PASTA_ACESSOS, f"{novo_nome}.png")
            if os.path.exists(foto_atual):
                os.rename(foto_atual, nova_foto)

            cursor.execute("""
                UPDATE pessoas_autorizadas
                SET nome = ?, foto_arquivo = ?
                WHERE id_pessoa = ?;
            """, (novo_nome, nova_foto, id_p))
            conexao.commit()
            print("Nome e foto atualizados com sucesso!")

    elif opcao == "2":
        capturar_foto_webcam(nome_atual)
        print("Nova foto salva com sucesso!")


def remover_acesso():
    usuarios = listar_usuarios()
    if not usuarios:
        return

    id_escolhido = input("\nDigite o ID do usuário que deseja remover: ").strip()

    cursor.execute("""
        SELECT nome, foto_arquivo
        FROM pessoas_autorizadas
        WHERE id_pessoa = ?;
    """, (id_escolhido,))
    usuario = cursor.fetchone()

    if not usuario:
        print("Usuário não encontrado.")
        return

    nome, foto_arquivo = usuario

    cursor.execute("""
        DELETE FROM pessoas_autorizadas
        WHERE id_pessoa = ?;
    """, (id_escolhido,))
    conexao.commit()

    if os.path.exists(foto_arquivo):
        os.remove(foto_arquivo)

    print(f"Usuário '{nome}' e seus registros foram apagados (ON DELETE CASCADE)!")


def listar_historico_acessos():
    usuarios = listar_usuarios()
    if not usuarios:
        return

    id_escolhido = input("\nDigite o ID (PK) do usuário para consultar seus acessos: ").strip()

    cursor.execute("""
        SELECT id_pessoa, nome
        FROM pessoas_autorizadas
        WHERE id_pessoa = ?;
    """, (id_escolhido,))
    pessoa = cursor.fetchone()

    if not pessoa:
        print("Usuário não encontrado na tabela pessoas_autorizadas.")
        return

    id_pk, nome_pessoa = pessoa

    cursor.execute("""
        SELECT id_registro, id_pessoa, data_hora
        FROM registros_acessos
        WHERE id_pessoa = ?
        ORDER BY id_registro DESC;
    """, (id_pk,))
    logs = cursor.fetchall()

    print(f"\n--- TABELA FILHA: HISTÓRICO DE ACESSOS DE {nome_pessoa.upper()} ---")
    print(f"(Buscando onde FOREIGN KEY id_pessoa = {id_pk})")

    if not logs:
        print("Nenhum registro de acesso encontrado para este usuário.")
    else:
        for id_reg, fk_pessoa, dt_hora in logs:
            print(f"ID Registro (PK): {id_reg} | ID Pessoa (FK): {fk_pessoa} | Data/Hora: {dt_hora}")


def acessar_sistema():
    """Abre a webcam, reconhece o rosto com limiar flexível e salva o log."""
    reconhecedor, mapa_nomes = treinar_reconhecedor()
    camera = cv2.VideoCapture(0)
    ultimo_log = {}

    print("\n[PORTARIA ATIVA] Pressione ESPAÇO na janela da câmera para sair.")

    while True:
        sucesso, frame = camera.read()
        if not sucesso:
            break

        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostos = detector_rosto.detectMultiScale(cinza, scaleFactor=1.3, minNeighbors=5)

        for (x, y, l, a) in rostos:
            autorizado = False
            nome_pessoa = ""
            id_p = None
            distancia_str = ""

            if reconhecedor is not None and len(mapa_nomes) > 0:
                rosto_atual = cv2.resize(cinza[y:y+a, x:x+l], (200, 200))
                id_predito, distancia = reconhecedor.predict(rosto_atual)
                distancia_str = f" (Dist: {int(distancia)})"

                # Limiar ajustado para 120 (mais tolerante para webcams simples de notebook)
                if distancia < 120 and id_predito in mapa_nomes:
                    autorizado = True
                    id_p = id_predito
                    nome_pessoa = mapa_nomes[id_predito]

            if autorizado:
                agora = datetime.now()
                data_hora_str = agora.strftime("%d/%m/%Y %H:%M:%S")

                if id_p not in ultimo_log or (agora - ultimo_log[id_p]).seconds >= 10:
                    cursor.execute("""
                        INSERT INTO registros_acessos (id_pessoa, data_hora)
                        VALUES (?, ?);
                    """, (id_p, data_hora_str))
                    conexao.commit()
                    ultimo_log[id_p] = agora
                    print(f"[LOG SALVO] Entrada de {nome_pessoa} autorizada - {data_hora_str}")

                cor = (0, 255, 0)  # Verde
                mensagem = f"Autorizado: {nome_pessoa}{distancia_str}"
            else:
                cor = (0, 0, 255)  # Vermelho
                mensagem = f"Nao autorizado{distancia_str}"

            cv2.rectangle(frame, (x, y), (x + l, y + a), cor, 2)
            cv2.putText(frame, mensagem, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2)

        cv2.imshow("Controle de Acesso Facial (ESPACO para sair)", frame)

        if cv2.waitKey(1) & 0xFF == 32:
            break

    camera.release()
    cv2.destroyAllWindows()


# =====================================================================
# 5. LOGIN E MENU PRINCIPAL EM MODO TEXTO
# =====================================================================
print("=== SISTEMA DE CONTROLE DE ACESSO FACIAL ===")

while True:
    senha_digitada = input("Digite a senha de administrador: ").strip()
    if senha_digitada == SENHA_ADMIN:
        print("Admin autorizado")
        break
    else:
        print("Senha incorreta! Tente novamente.")

while True:
    print("\n---------------- MENU ----------------")
    print("1 - Cadastrar acesso")
    print("2 - Editar acesso")
    print("3 - Remover acesso")
    print("4 - Acessar (Reconhecimento Facial)")
    print("5 - Listar histórico de acessos (Relacionamento PK -> FK)")
    print("6 - Sair")
    opcao_menu = input("Escolha uma opção: ").strip()

    if opcao_menu == "1":
        cadastrar_acesso()
    elif opcao_menu == "2":
        editar_acesso()
    elif opcao_menu == "3":
        remover_acesso()
    elif opcao_menu == "4":
        acessar_sistema()
    elif opcao_menu == "5":
        listar_historico_acessos()
    elif opcao_menu == "6":
        print("Encerrando o sistema...")
        break
    else:
        print("Opção inválida! Tente novamente.")

conexao.close()