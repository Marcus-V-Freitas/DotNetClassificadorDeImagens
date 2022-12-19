using DotNetClassificadorDeImagens.Classes;
using DotNetClassificadorDeImagens.Extension;

namespace DotNetClassificadorDeImagens;

public sealed class Program
{
    private const string IMAGENS = "..\\..\\..\\Data";
    private const string MODELO = "..\\..\\..\\Model\\animals.zip";
    private const string COLUNA_RESULTADOS = "PredictedLabel";
    private const string COLUNA_ROTULOS = "LabelAsKey";

    public static void Main(string[] args)
    {
        var classificadorImagens = new ClassificadorDeImagens(IMAGENS,
                                                              COLUNA_ROTULOS,
                                                              COLUNA_RESULTADOS);

        var dadosTreinamentoTeste = classificadorImagens.DivisaoDeTreinamentoTeste();
        var dadosTreinamento = dadosTreinamentoTeste.TrainSet;
        var dadosTeste = dadosTreinamentoTeste.TestSet;

        var modelo = classificadorImagens.Treinar(dadosTreinamento, dadosTeste);
        var metricas = classificadorImagens.Pontuacao(modelo, dadosTeste);

        metricas.ExibirMetricas();

        classificadorImagens.SalvarModelo(modelo, dadosTreinamento.Schema, MODELO);

        Console.ReadKey();
    }
}