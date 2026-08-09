"""Testes da Task 1: Estrutura do projeto e Docker."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def test_project_structure_exists():
    """Estrutura de pastas deve existir."""
    assert (ROOT / "app").is_dir()
    assert (ROOT / "app" / "agents").is_dir()
    assert (ROOT / "app" / "agents" / "docs").is_dir()
    assert (ROOT / "app" / "agents" / "finance").is_dir()
    assert (ROOT / "app" / "core").is_dir()
    assert (ROOT / "app" / "embeddings").is_dir()
    assert (ROOT / "app" / "extraction").is_dir()
    assert (ROOT / "app" / "llm").is_dir()
    assert (ROOT / "app" / "orchestrator").is_dir()
    assert (ROOT / "app" / "rag").is_dir()
    assert (ROOT / "app" / "api").is_dir()
    assert (ROOT / "tests").is_dir()
    assert (ROOT / "tests" / "unit").is_dir()
    assert (ROOT / "tests" / "integration").is_dir()
    assert (ROOT / "tests" / "e2e").is_dir()
    assert (ROOT / "scripts").is_dir()


def test_required_files_exist():
    """Arquivos de configuração devem existir."""
    assert (ROOT / "requirements.txt").exists()
    assert (ROOT / "docker-compose.yml").exists()
    assert (ROOT / ".env.example").exists()
    assert (ROOT / "pyproject.toml").exists()
    assert (ROOT / ".gitignore").exists()


def test_docker_compose_valid():
    """docker-compose.yml deve ser válido."""
    result = subprocess.run(
        ["docker", "compose", "config"],
        capture_output=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_env_example_has_required_vars():
    """.env.example deve ter as variáveis obrigatórias."""
    content = (ROOT / ".env.example").read_text()
    required = [
        "DATABASE_URL",
        "OLLAMA_BASE_URL",
        "EMBEDDING_MODEL",
        "LLM_MODEL",
    ]
    for var in required:
        assert var in content, f"Variável {var} não encontrada em .env.example"
