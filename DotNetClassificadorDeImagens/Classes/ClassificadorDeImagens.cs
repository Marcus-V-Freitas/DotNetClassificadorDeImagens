using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;
using static Microsoft.ML.DataOperationsCatalog;

namespace DotNetClassificadorDeImagens.Classes;


/// <summary>
/// Classe wrapper para classificar as imagens
/// </summary>
public sealed class ClassificadorDeImagens
{
    private readonly MLContext _contexto;
    private readonly string _nomeColunaRotulos;
    private readonly string _nomeDaColunaDeResultados;
    private readonly IDataView _imagens;

    /// <summary>
    /// Inicializa uma nova instância <see cref="ClassificadorDeImagens"/> class.
    /// </summary>
    /// <param name="caminhoDasImagens"> Caminho das imagens </param>
    /// <param name="nomeColunaRotulos"> Nome da coluna de rótulos </param>
    /// <param name="nomeDaColunaDeResultados"> Nome da coluna de resultados previstos </param>
    /// <param name="seed"> Seed </param>
    public ClassificadorDeImagens(string caminhoDasImagens, string nomeColunaRotulos, string nomeDaColunaDeResultados, int seed = 0)
    {
        _contexto = new(seed);
        _nomeColunaRotulos = nomeColunaRotulos;
        _nomeDaColunaDeResultados = nomeDaColunaDeResultados;

        var dados = ExtrairDadosDasImagens(caminhoDasImagens);

        _imagens = CarregarRotulosDasImagens(dados, caminhoDasImagens);
    }

    /// <summary>
    /// Extraí os dados das imagens para a classificação
    /// </summary>
    /// <param name="caminhoDasImagens"> Caminho das imagens </param>
    /// <returns> Dados de imagem </returns>
    private IDataView ExtrairDadosDasImagens(string caminhoDasImagens)
    {
        var entradas = CarregarDadosBaseDasImagens(caminhoDasImagens);
        var dados = _contexto.Data.LoadFromEnumerable(data: entradas);
        dados = _contexto.Data.ShuffleRows(input: dados);

        return dados;
    }

    /// <summary>
    /// Cria o pipeline para geração do modelo
    /// </summary>
    /// <param name="opcoes"> opções de definição do pipeline </param>
    /// <returns> Pipeline de estimação </returns>
    private EstimatorChain<KeyToValueMappingTransformer> CriarPipeline(ImageClassificationTrainer.Options opcoes)
    {
        var pipeline = _contexto.MulticlassClassification.Trainers.ImageClassification(options: opcoes)
           .Append(_contexto.Transforms.Conversion.MapKeyToValue(_nomeDaColunaDeResultados));

        return pipeline;
    }

    /// <summary>
    /// Define os parâmetros do pipeline para criação do modelo
    /// </summary>
    /// <param name="dadosTestes"> Dados de testes </param>
    /// <returns> Opções de pipeline </returns>
    private ImageClassificationTrainer.Options DefinirOpcoesDePipeline(IDataView dadosTestes)
    {
        var opcoes = new ImageClassificationTrainer.Options()
        {
            FeatureColumnName = nameof(Entrada.Imagem),
            LabelColumnName = _nomeColunaRotulos,
            ValidationSet = dadosTestes,
            Arch = ImageClassificationTrainer.Architecture.ResnetV2101, // DNN pré-treinado
            MetricsCallback = (metricas) => Console.WriteLine(metricas),
            TestOnTrainSet = false
        };

        return opcoes;
    }

    /// <summary>
    /// Carrega as imagens e converte os rótulos em chaves para servir como valores categóricos
    /// </summary>
    /// <param name="dados"> dados base das imagens </param>
    /// <param name="caminhoDasImagens"> caminho das imagens </param>
    /// <returns> Dados das imagens </returns>
    private IDataView CarregarRotulosDasImagens(IDataView dados, string caminhoDasImagens)
    {
        var imagens = _contexto.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(Entrada.Rotulo),
                                                                    outputColumnName: _nomeColunaRotulos)
            .Append(_contexto.Transforms.LoadRawImageBytes(inputColumnName: nameof(Entrada.CaminhoImagem),
                                                           outputColumnName: nameof(Entrada.Imagem),
                                                           imageFolder: caminhoDasImagens))
            .Fit(input: dados)
            .Transform(input: dados);

        return imagens;
    }

    /// <summary>
    /// Carrega informações básicas das imagens como caminho e nome do diretorio (label)
    /// </summary>
    /// <param name="caminhoDasImagens"> Caminho das imagens </param>
    /// <returns> Lista de dados básicos das imagens </returns>
    private List<Entrada> CarregarDadosBaseDasImagens(string caminhoDasImagens)
    {
        var imagens = new List<Entrada>();
        var diretorios = Directory.EnumerateDirectories(caminhoDasImagens);

        foreach (var diretorio in diretorios)
        {
            var arquivos = Directory.EnumerateFiles(diretorio);

            imagens.AddRange(arquivos.Select(imagem => new Entrada
            {
                CaminhoImagem = Path.GetFullPath(imagem),
                Rotulo = Path.GetFileName(diretorio)
            }));
        }

        return imagens;
    }

    /// <summary>
    /// Realiza a criação do modelo 
    /// </summary>
    /// <param name="dadosTreinamento"> Dados de treinamento </param>
    /// <param name="dadosTestes"> Dados de testes </param>
    /// <returns> Modelo de classificação de imagens </returns>
    public TransformerChain<KeyToValueMappingTransformer> Treinar(IDataView dadosTreinamento, IDataView dadosTestes)
    {
        var opcoes = DefinirOpcoesDePipeline(dadosTestes);

        var pipeline = CriarPipeline(opcoes);

        Console.WriteLine("Treinando o modelo...");

        var modelo = pipeline.Fit(input: dadosTreinamento);

        Console.WriteLine("Treinamento do modelo concluído!");

        return modelo;
    }

    /// <summary>
    /// Gera as previsões dos resultados dos dados de teste
    /// </summary>
    /// <param name="modelo"> modelo de classificação de imagens </param>
    /// <param name="dadosTeste"> Dados de testes </param>
    /// <returns> Dados previstos </returns>
    public IDataView Prever(TransformerChain<KeyToValueMappingTransformer> modelo, IDataView dadosTeste)
    {
        return modelo.Transform(input: dadosTeste);
    }

    /// <summary>
    /// Pontuacaoes the specified modelo.
    /// </summary>
    /// <param name="modelo"> Modelo de classificação de imagens </param>
    /// <param name="dadosTeste"> Dados de testes </param>
    /// <returns> Métricas do conjunto de teste </returns>
    public MulticlassClassificationMetrics Pontuacao(TransformerChain<KeyToValueMappingTransformer> modelo, IDataView dadosTeste)
    {
        var previsoes = Prever(modelo, dadosTeste);

        return _contexto.MulticlassClassification.Evaluate(data: previsoes,
                                                           labelColumnName: _nomeColunaRotulos,
                                                           predictedLabelColumnName: _nomeDaColunaDeResultados);
    }


    /// <summary>
    /// Divide o conjunto em dados de treinamento e teste
    /// </summary>
    /// <param name="fracaoDeTeste"> Fração do conjunto dedicada aos dados de teste </param>
    /// <param name="seed"> Seed </param>
    /// <returns> Conjuntos de treinamento e teste </returns>
    public TrainTestData DivisaoDeTreinamentoTeste(double fracaoDeTeste = 0.2, int seed = 1)
    {
        return _contexto.Data.TrainTestSplit(data: _imagens,
                                             testFraction: fracaoDeTeste,
                                             seed: seed);
    }

    /// <summary>
    /// Salva o modelo gerado baseado no schema no diretório informado
    /// </summary>
    /// <param name="modelo"> Modelo de classificação de imagens </param>
    /// <param name="schema"> Schema do modelo </param>
    /// <param name="caminhoDeSalvamento"> Caminho de salvamento do modelo </param>
    public void SalvarModelo(TransformerChain<KeyToValueMappingTransformer> modelo, DataViewSchema schema, string caminhoDeSalvamento)
    {
        if (!Directory.Exists(caminhoDeSalvamento))
        {
            Directory.CreateDirectory(caminhoDeSalvamento);
        }

        Console.WriteLine("Salvando o modelo...");

        _contexto.Model.Save(model: modelo,
                             inputSchema: schema,
                             filePath: caminhoDeSalvamento);

        Console.WriteLine("Modelo salvo com sucesso!");
    }
}