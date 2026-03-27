"""
图数据库工具类
"""
from typing import Optional, Dict, Any
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from src.config.settings import API_CONFIG


class GraphDBManager:
    """图数据库管理器"""
    
    def __init__(self, database: Optional[str] = None):
        self.database = database or API_CONFIG["neo4j_database"]
        self.graph = None
        self.chain = None
        self.llm = None
        self._initialize_graph()
    
    def _initialize_graph(self):
        """初始化图数据库连接"""
        try:
            self.graph = Neo4jGraph(database=self.database)
            self.llm = ChatOpenAI(
                model=API_CONFIG["model_name"],
                temperature=0,
                base_url=API_CONFIG["base_url"],
                api_key=API_CONFIG["api_key"]
            )
            self._create_cypher_chain()
            print(f"✅ 图数据库连接成功: {self.database}")
        except Exception as e:
            print(f"❌ 图数据库连接失败: {e}")
            print("将使用替代方案进行图数据查询")
            self.graph = None
            self.chain = None
    
    def _create_cypher_chain(self):
        """创建Cypher查询链"""
        if self.graph is None:
            return
        
        CYPHER_GENERATION_TEMPLATE = """
        你是一个 Neo4j 专家，任务是将自然语言问题转换为 Cypher 查询。
        图数据库的 schema 如下：
        {schema}

        请严格遵守以下规则：
        1. **节点匹配必须使用 `id` 属性进行精确匹配**，例如：`(n:皮肤病 {{id: "扁平疣"}})`
        2. 不要使用 `CONTAINS`、`=~` 或其他模糊匹配。
        3. 只返回 Cypher 查询语句，不要解释，不要 markdown，不要反引号。
        4. 对于病症遵循以下规则：皮肤病-[辨证为]->证型-[主症包括]->症状
        5. 证型-[治法为]->方剂-[用于治疗]->皮肤病
        问题：{question}
        """

        cypher_prompt = PromptTemplate(
            template=CYPHER_GENERATION_TEMPLATE,
            input_variables=["schema", "question"]
        )

        self.chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            llm=self.llm,
            cypher_prompt=cypher_prompt,
            verbose=True,
            allow_dangerous_requests=True,
            validate_cypher=True,
            fix_cypher=True,
            max_fix_attempts=2,
        )
    
    def query(self, question: str) -> str:
        """查询图数据库"""
        if self.chain is None:
            return "图数据库暂时不可用，将使用通用模型进行回答。"
        
        try:
            response = self.chain.invoke({"query": question})
            return response.get("result", "未找到相关信息")
        except Exception as e:
            return f"图数据库查询出错: {str(e)}"
    
    def is_available(self) -> bool:
        """检查图数据库是否可用"""
        return self.graph is not None and self.chain is not None