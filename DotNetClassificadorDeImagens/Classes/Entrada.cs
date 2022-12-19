namespace DotNetClassificadorDeImagens.Classes;

public sealed class Entrada
{
    public byte[] Imagem { get; set; }
    public string CaminhoImagem { get; set; }
    public string Rotulo { get; set; }
}