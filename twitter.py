import asyncio
import os
from twikit import Client, Notification, Tweet
from grafo import ejecutar_grafo

#Variables de entorno para las credenciales de Twitter
username = os.getenv("TWITTER_USERNAME")  # O reemplaza con tu nombre de usuario
mail = os.getenv("TWITTER_EMAIL")  # O reemplaza con tu correo electrónico
password = os.getenv("TWITTER_PASSWORD")  # O reemplaza con tu contraseña
cookies_file = "cookies.json"  # Archivo para almacenar cookies de sesión

#Funcion para logear la cuenta de Twitter usando Twikit

async def logear_cuenta():
    cliente=Client('es-ES') # Español de España
    await cliente.login(
        auth_info_1=username,
        auth_info_2=mail, #opcional pero recomendable
        password=password,
        cookies_file=cookies_file
    )
    print("Cuenta logeada exitosamente")
    return cliente

async def logout_cuenta(cliente):
    await cliente.logout()
    print("Cuenta cerrada exitosamente")

# 
async def detectar_menciones_nuevas(cliente: Client, ultimo_tweet_id: int=None) -> list[Notification]:
    menciones=await cliente.get_notifications(type="mention",count=40) # Obtener las últimas 40 menciones
    nuevas_menciones=[]

    for mencion in menciones: # Recorremos las menciones
        if ultimo_tweet_id is None or mencion.tweet.id > ultimo_tweet_id:
            nuevas_menciones.append(mencion)
        else:
            break # Estan ordenadas de mas recientes a mas antiguas
    
    nuevas_menciones.reverse() # Para procesarlas de la más antigua a la más reciente
    return nuevas_menciones

async def procesar_mencion(cliente: Client, mencion: Tweet):
    texto=mencion.text
    perfil_usuario=mencion.user.screen_name # Nombre de usuario sin @

    respuesta=await ejecutar_grafo(texto, perfil_usuario)
    return respuesta
     



async def main():
    client = await logear_cuenta() # Logeamos la cuenta una sola vez al inicio
    nuevo_ultimo_id = None 
    try:
        while True:
            try:

                nuevas_menciones = await detectar_menciones_nuevas(client,nuevo_ultimo_id) # Detectamos nuevas menciones desde el último ID conocido
                for mencion in nuevas_menciones:
                    print(f"Nueva mención de @{mencion.from_user}: {mencion.tweet.text}")
                    respuesta = await procesar_mencion(client, mencion)
                    await client.create_tweet(
                        text=respuesta,
                        reply_to=mencion.tweet.id
                    )
                    print(f"Respuesta enviada a @{mencion.from_user}")

            
                if nuevas_menciones:
                    nuevo_ultimo_id = nuevas_menciones[-1].id # Actualizamos el último ID conocido al más reciente procesado


                

            except Exception as e:
                print(f"Error al detectar menciones: {e}")
            finally:
                await asyncio.sleep(30) # Esperamos 30 segundos antes de volver a revisar

    except KeyboardInterrupt:
        print("Interrupción manual recibida. Cerrando sesión...")

    finally:
        await logout_cuenta(client) # Cerramos la sesión al finalizar

if __name__ == "__main__":
    asyncio.run(main())