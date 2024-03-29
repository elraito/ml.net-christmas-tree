using Microsoft.Extensions.ML;
using Microsoft.ML;
using Server;
using Server.Models;
using Server.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.AddSingleton(sp => new MLContext());
builder.Services.AddSingleton<IModelBuilderService, ModelBuilderService>();
builder.Services.AddSingleton<IInferenceService, InferenceService>();

builder.Services.AddPredictionEnginePool<ModelInput, ModelOutput>()
    .FromFile("/data/workspace/model.zip");

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseDefaultFiles();
app.UseStaticFiles();

app.MapModelBuilderRoutes();


app.Run();
